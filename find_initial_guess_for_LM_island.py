import datetime
import os
import pandas as pd
import numpy as np
import QuantLib as ql
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import time
import multiprocessing
import threading
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FOLDER_ZERO_CURVES: str = os.path.join(DATA_DIR, 'EUR ZERO CURVE')
FOLDER_VOLATILITY_CUBES: str = os.path.join(DATA_DIR, 'EUR BVOL CUBE')
OUTPUT_DIR: str = os.path.join(RESULTS_DIR, 'initial_guess_analysis_ga_extended_hw')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HELPER FUNCTIONS (INCLUDES HELPERS FROM LM SCRIPT) ---
def parse_tenor(tenor_str: str) -> ql.Period:
    tenor_str = tenor_str.strip().upper()
    if 'YR' in tenor_str: num = int(tenor_str.replace('YR', '')); return ql.Period(num, ql.Years)
    if 'MO' in tenor_str: num = int(tenor_str.replace('MO', '')); return ql.Period(num, ql.Months)
    raise ValueError(f"Could not parse tenor string: {tenor_str}")

def parse_tenor_to_years(tenor_str: str) -> float:
    tenor_str = tenor_str.strip().upper()
    if 'YR' in tenor_str: return float(int(tenor_str.replace('YR', '')))
    if 'MO' in tenor_str: return int(tenor_str.replace('MO', '')) / 12.0
    raise ValueError(f"Could not parse tenor string to years: {tenor_str}")

def _get_step_dates_from_expiries(
    ql_eval_date: ql.Date,
    included_expiries_yrs: List[float],
    num_segments: int
) -> List[ql.Date]:
    if num_segments <= 1: return []
    unique_expiries = sorted(list(set(included_expiries_yrs)))
    if len(unique_expiries) < num_segments:
         print(f"Warning: Not enough unique expiries ({len(unique_expiries)}) to create {num_segments - 1} steps. Reducing number of segments.")
         num_segments = len(unique_expiries)
    if num_segments <= 1: return []
    # Select num_segments-1 points that divide the expiries into num_segments partitions
    indices = np.linspace(0, len(unique_expiries) - 1, num_segments + 1).astype(int)[1:-1]
    time_points_in_years = [unique_expiries[i] for i in indices]
    step_dates = [ql_eval_date + ql.Period(int(y * 365.25), ql.Days) for y in time_points_in_years]
    return step_dates

def create_ql_yield_curve(zero_curve_df: pd.DataFrame, eval_date: datetime.date) -> ql.RelinkableYieldTermStructureHandle:
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    dates = [ql_eval_date] + [ql.Date(d.day, d.month, d.year) for d in pd.to_datetime(zero_curve_df['Date'])]
    rates = [zero_curve_df['ZeroRate'].iloc[0]] + zero_curve_df['ZeroRate'].tolist()
    term_structure = ql.ZeroCurve(dates, rates, ql.Actual365Fixed(), ql.TARGET(), ql.Linear(), ql.Continuous, ql.Annual)
    term_structure.enableExtrapolation()
    handle = ql.RelinkableYieldTermStructureHandle()
    handle.linkTo(term_structure)
    return handle

def prepare_calibration_helpers(vol_cube_df: pd.DataFrame, term_structure_handle: ql.RelinkableYieldTermStructureHandle, min_expiry_years: float, min_tenor_years: float) -> List[ql.SwaptionHelper]:
    helpers = []
    vols_df = vol_cube_df[vol_cube_df['Type'] == 'Vol'].set_index('Expiry')
    swap_index = ql.Euribor6M(term_structure_handle)
    for expiry_str in vols_df.index:
        for tenor_str in vols_df.columns:
            if tenor_str == 'Type': continue
            vol = vols_df.loc[expiry_str, tenor_str]
            if pd.isna(vol): continue
            if parse_tenor_to_years(expiry_str) < min_expiry_years or parse_tenor_to_years(tenor_str) < min_tenor_years: continue
            vol_handle = ql.QuoteHandle(ql.SimpleQuote(vol / 10000.0))
            helper = ql.SwaptionHelper(parse_tenor(expiry_str), parse_tenor(tenor_str), vol_handle, swap_index, ql.Period(6, ql.Months), swap_index.dayCounter(), swap_index.dayCounter(), term_structure_handle, ql.Normal)
            helpers.append(helper)
    return helpers

def progress_listener(queue: multiprocessing.Queue, total: int):
    """Listens to the queue and updates the TQDM progress bar."""
    pbar = tqdm(total=total, desc="Epoch Generations")
    while True:
        message = queue.get()
        if message == "STOP":
            break
        pbar.update(1)
    pbar.close()

# --- ISLAND MODEL WORKER FUNCTION (EXTENDED FOR 1+7 PARAMETERS) ---
def evolve_island(args):
    """
    Runs a self-contained GA on a single island to optimize a 1+7 parameter Hull-White model.
    """
    (island_id, population, generations_per_epoch, ga_params, ql_init_data, progress_queue) = args
    (bounds, mutation_rate, tournament_size, elitism_count, mut_strength, num_sigma_segments) = ga_params
    (eval_date_dt, zero_df, vol_df, min_exp, min_ten) = ql_init_data

    ql_eval_date = ql.Date(eval_date_dt.day, eval_date_dt.month, eval_date_dt.year)
    ql.Settings.instance().evaluationDate = ql_eval_date
    random.seed(os.getpid() * int(time.time()) % 123456789)
    np.random.seed(os.getpid() * int(time.time()) % 123456789)
    
    term_structure = create_ql_yield_curve(zero_df, eval_date_dt)
    helpers = prepare_calibration_helpers(vol_df, term_structure, min_exp, min_ten)
    
    # --- START OF THE FIX ---
    # Determine step dates for piecewise volatility ONCE, as they are static for this process
    # Correctly calculate years to expiry from the helper's date objects
    day_counter = ql.Actual365Fixed()
    included_expiries_yrs = sorted(list(set([
        day_counter.yearFraction(ql_eval_date, h.swaption().exercise().dates()[-1]) for h in helpers
    ])))
    # --- END OF THE FIX ---
    
    sigma_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, num_sigma_segments)

    fitness_cache = {}
    def calculate_rmse_local(params: Tuple):
        if params in fitness_cache:
            return fitness_cache[params]
        
        a_val = params[0]
        sigma_vals = params[1:]

        reversion_handle = [ql.QuoteHandle(ql.SimpleQuote(a_val))]
        sigma_handles = [ql.QuoteHandle(ql.SimpleQuote(s)) for s in sigma_vals]
        
        model = ql.Gsr(term_structure, sigma_step_dates, sigma_handles, reversion_handle)
        engine = ql.Gaussian1dSwaptionEngine(model)
        
        squared_errors = []
        for helper in helpers:
            helper.setPricingEngine(engine)
            market_vol_bps = helper.volatility().value() * 10000
            try:
                model_npv = helper.modelValue()
                implied_vol_bps = helper.impliedVolatility(model_npv, 1.0e-4, 100, 1.0e-7, 0.5) * 10000
                squared_errors.append((implied_vol_bps - market_vol_bps)**2)
            except RuntimeError: continue
        rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else float('inf')
        fitness_cache[params] = rmse
        return rmse

    # ... (rest of the function is unchanged) ...
    for gen in range(generations_per_epoch):
        fitness_values = [calculate_rmse_local(ind) for ind in population]
        pop_with_fitness = sorted(zip(population, fitness_values), key=lambda x: x[1])
        next_generation = [ind for ind, fit in pop_with_fitness[:elitism_count]]
        
        while len(next_generation) < len(population):
            p1 = min(random.sample(pop_with_fitness, tournament_size), key=lambda x: x[1])[0]
            p2 = min(random.sample(pop_with_fitness, tournament_size), key=lambda x: x[1])[0]
            
            child = tuple([(p1[i] + p2[i]) / 2.0 for i in range(len(p1))])
            
            if random.random() < mutation_rate:
                mutated_params = list(child)
                a_range = bounds['a'][1] - bounds['a'][0]
                mut_std_a = (a_range * 0.1) * mut_strength
                mutated_params[0] = np.clip(mutated_params[0] + random.gauss(0, mut_std_a), *bounds['a'])
                
                sigma_range = bounds['sigma'][1] - bounds['sigma'][0]
                mut_std_sigma = (sigma_range * 0.1) * mut_strength
                for i in range(1, len(mutated_params)):
                    mutated_params[i] = np.clip(mutated_params[i] + random.gauss(0, mut_std_sigma), *bounds['sigma'])
                
                child = tuple(mutated_params)
            next_generation.append(child)
        population = next_generation
        progress_queue.put(1)

    final_fitness_values = [calculate_rmse_local(ind) for ind in population]
    final_pop_with_fitness = sorted(zip(population, final_fitness_values), key=lambda x: x[1])
    
    return final_pop_with_fitness, fitness_cache

# --- VISUALIZATION FUNCTIONS (ADAPTED FROM LM SCRIPT) ---
def plot_calibration_results(results_df: pd.DataFrame, eval_date: datetime.date, save_dir: str):
    plot_data = results_df.dropna(subset=['MarketVol', 'ModelVol', 'Difference_bps']).copy()
    if plot_data.empty: return
    X = plot_data['Expiry'].values
    Y = plot_data['Tenor'].values
    Z_market, Z_model, Z_diff = plot_data['MarketVol'].values, plot_data['ModelVol'].values, plot_data['Difference_bps'].values
    
    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(f'GA Hull-White Calibration Volatility Surfaces for {eval_date}', fontsize=16)
    
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.set_title('Observed Market Volatilities (bps)')
    surf1 = ax1.plot_trisurf(X, Y, Z_market, cmap=cm.viridis, antialiased=True, linewidth=0.1)
    ax1.set_xlabel('Expiry (Years)'); ax1.set_ylabel('Tenor (Years)'); ax1.set_zlabel('Volatility (bps)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
    ax1.view_init(elev=30, azim=-120)

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.set_title('Model Implied Volatilities (bps)')
    surf2 = ax2.plot_trisurf(X, Y, Z_model, cmap=cm.viridis, antialiased=True, linewidth=0.1)
    ax2.set_xlabel('Expiry (Years)'); ax2.set_ylabel('Tenor (Years)'); ax2.set_zlabel('Volatility (bps)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
    ax2.set_zlim(np.nanmin(Z_market) * 0.9, np.nanmax(Z_market) * 1.1)
    ax2.view_init(elev=30, azim=-120)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.set_title('Difference (Model - Market) (bps)')
    surf3 = ax3.plot_trisurf(X, Y, Z_diff, cmap=cm.coolwarm, antialiased=True, linewidth=0.1)
    ax3.set_xlabel('Expiry (Years)'); ax3.set_ylabel('Tenor (Years)'); ax3.set_zlabel('Difference (bps)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.1)
    ax3.set_zlim(-np.nanmax(np.abs(Z_diff)), np.nanmax(np.abs(Z_diff)))
    ax3.view_init(elev=30, azim=-120)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f'GA_CalibrationPlot_{eval_date}.png'))
    plt.close(fig)
    print(f"\nGA calibration surface plot saved to: {os.path.join(save_dir, f'GA_CalibrationPlot_{eval_date}.png')}")

# --- MAIN WORKFLOW ---
if __name__ == '__main__':
    # --- GA and Model Hyperparameters ---
    NUM_SIGMA_SEGMENTS = 7
    NUM_ISLANDS = multiprocessing.cpu_count()
    MIGRATION_INTERVAL = 15
    MIGRATION_COUNT = 2
    TOTAL_POPULATION = 512  # Increased for higher dimensional space
    TOTAL_GENERATIONS = 150
    MUTATION_RATE = 0.9     # Higher mutation rate is often better for complex spaces
    TOURNAMENT_SIZE = 3
    ELITISM_COUNT = 1
    INITIAL_MUTATION_STRENGTH = 1.0
    FINAL_MUTATION_STRENGTH = 0.05
    
    POP_PER_ISLAND = max(10, TOTAL_POPULATION // NUM_ISLANDS)
    NUM_EPOCHS = TOTAL_GENERATIONS // MIGRATION_INTERVAL

    PARAM_BOUNDS = {'a': (0.005, 0.05), 'sigma': (0.00001, 0.001)} # Sigma bounds apply to all 7 sigmas
    MIN_EXPIRY_YEARS, MIN_TENOR_YEARS = 2.0, 2.0
    date_str = "03.08.2025"
    eval_date = datetime.datetime.strptime(date_str, "%d.%m.%Y").date()
    
    print(f"--- Starting GA for Extended HW (1+7) model for {date_str} ---")
    print(f"Configuration: {NUM_ISLANDS} islands, {POP_PER_ISLAND} pop/island, {NUM_EPOCHS} epochs.")

    try:
        vol_files_folder = os.path.join(FOLDER_VOLATILITY_CUBES, 'xlsx')
        zero_curve_path = os.path.join(FOLDER_ZERO_CURVES, f"{date_str}.csv")
        vol_cube_path = os.path.join(vol_files_folder, f"{date_str}.xlsx")
        zero_curve_df = pd.read_csv(zero_curve_path)
        vol_cube_df = pd.read_excel(vol_cube_path, engine='openpyxl')
        vol_cube_df.rename(columns={'Unnamed: 1': 'Type'}, inplace=True)
        vol_cube_df['Expiry'] = vol_cube_df['Expiry'].ffill()

        ql_init_data = (eval_date, zero_curve_df, vol_cube_df, MIN_EXPIRY_YEARS, MIN_TENOR_YEARS)
        start_time = time.monotonic()
        
        # Create initial populations with 8 parameters per individual
        island_populations = []
        for _ in range(NUM_ISLANDS):
            island_pop = []
            for _ in range(POP_PER_ISLAND):
                a = random.uniform(*PARAM_BOUNDS['a'])
                sigmas = [random.uniform(*PARAM_BOUNDS['sigma']) for _ in range(NUM_SIGMA_SEGMENTS)]
                island_pop.append(tuple([a] + sigmas))
            island_populations.append(island_pop)

        best_overall_individual, best_overall_fitness = None, float('inf')
        fitness_history = []
        global_fitness_cache = {}

        # --- START: CORRECTED MAIN LOOP WITH MANAGER ---
        # Create a Manager to handle the shared queue
        with multiprocessing.Manager() as manager:
            progress_queue = manager.Queue()

            for epoch in range(NUM_EPOCHS):
                decay = (FINAL_MUTATION_STRENGTH / INITIAL_MUTATION_STRENGTH) ** (1/(NUM_EPOCHS-1)) if NUM_EPOCHS > 1 else 1.0
                mut_strength = INITIAL_MUTATION_STRENGTH * (decay ** epoch)
                ga_params = (PARAM_BOUNDS, MUTATION_RATE, TOURNAMENT_SIZE, ELITISM_COUNT, mut_strength, NUM_SIGMA_SEGMENTS)
                
                # The Manager's queue can be passed to the workers
                pool_args = [(i, island_populations[i], MIGRATION_INTERVAL, ga_params, ql_init_data, progress_queue) for i in range(NUM_ISLANDS)]

                print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} | Mutation Strength: {mut_strength:.4f} ---")
                
                total_generations_in_epoch = NUM_ISLANDS * MIGRATION_INTERVAL
                listener_thread = threading.Thread(target=progress_listener, args=(progress_queue, total_generations_in_epoch))
                listener_thread.start()
                
                # Create the Pool within the manager's context
                with multiprocessing.Pool(NUM_ISLANDS) as pool:
                    results = pool.map(evolve_island, pool_args)

                # Stop the listener thread
                progress_queue.put("STOP")
                listener_thread.join()

                # Process results and migrate (unchanged logic)
                island_pop_with_fitness = []
                for pop_with_fit, island_cache in results:
                    island_pop_with_fitness.append(pop_with_fit)
                    global_fitness_cache.update(island_cache)
                    if pop_with_fit and pop_with_fit[0][1] < best_overall_fitness:
                        best_overall_fitness = pop_with_fit[0][1]
                        best_overall_individual = pop_with_fit[0][0]

                fitness_history.append(best_overall_fitness)
                print(f"Epoch {epoch + 1} Best RMSE: {best_overall_fitness:.8f}")
                
                if epoch < NUM_EPOCHS - 1:
                    migrants = [pop[0:MIGRATION_COUNT] for pop in island_pop_with_fitness]
                    new_populations = []
                    for i in range(NUM_ISLANDS):
                        receiving_pop = [ind for ind, fit in island_pop_with_fitness[i]]
                        receiving_pop.sort(key=lambda ind: global_fitness_cache.get(ind, float('inf')), reverse=True)
                        donating_idx = (i - 1 + NUM_ISLANDS) % NUM_ISLANDS
                        migrating_inds = [ind for ind, fit in migrants[donating_idx]]
                        new_pop = receiving_pop[MIGRATION_COUNT:] + migrating_inds
                        random.shuffle(new_pop)
                        new_populations.append(new_pop)
                    island_populations = new_populations
        
        run_time = time.monotonic() - start_time
        print(f"\n--- GA Finished in {run_time:.2f} seconds ---")
        print(f"Total Unique Points Evaluated: {len(global_fitness_cache)}")
        print(f"Best RMSE found: {best_overall_fitness:.8f}")
        print(f"Best 'a': {best_overall_individual[0]:.8f}")
        for i in range(NUM_SIGMA_SEGMENTS):
            print(f"Best 'sigma_{i+1}': {best_overall_individual[i+1]:.8f}")

        initial_guess_dict = {'a': best_overall_individual[0], 'sigma': list(best_overall_individual[1:])}
        save_path_npy = os.path.join(OUTPUT_DIR, 'initial_guess_ga_extended.npy')
        np.save(save_path_npy, initial_guess_dict)
        print(f"\nBest initial guess saved to: {save_path_npy}")

        # --- Final Visualization ---
        ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
        term_structure_handle = create_ql_yield_curve(zero_curve_df, eval_date)
                
        helpers = prepare_calibration_helpers(vol_cube_df, term_structure_handle, MIN_EXPIRY_YEARS, MIN_TENOR_YEARS)
        day_counter = ql.Actual365Fixed()
        included_expiries_yrs = sorted(list(set([
            day_counter.yearFraction(ql_eval_date, h.swaption().exercise().dates()[-1]) for h in helpers
        ])))
        
        # Build final model for evaluation
        included_expiries_yrs = sorted(list(set([parse_tenor_to_years(h.swaption().exercise().dates()[-1].ISO()) for h in helpers])))
        sigma_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, NUM_SIGMA_SEGMENTS)
        final_model = ql.Gsr(term_structure_handle, sigma_step_dates, 
                             [ql.QuoteHandle(ql.SimpleQuote(s)) for s in best_overall_individual[1:]], 
                             [ql.QuoteHandle(ql.SimpleQuote(best_overall_individual[0]))])
        final_engine = ql.Gaussian1dSwaptionEngine(final_model)

        results_data = []
        for helper in helpers:
            helper.setPricingEngine(final_engine)
            market_vol_bps = helper.volatility().value() * 10000
            model_npv = helper.modelValue()
            model_vol_bps = helper.impliedVolatility(model_npv, 1.0e-4, 100, 1.0e-7, 0.5) * 10000
            results_data.append({
                'Expiry': parse_tenor_to_years(helper.swaption().exercise().dates()[-1].ISO()),
                'Tenor': parse_tenor_to_years(helper.swaption().maturityDate().ISO()),
                'MarketVol': market_vol_bps, 'ModelVol': model_vol_bps,
                'Difference_bps': model_vol_bps - market_vol_bps
            })
        results_df = pd.DataFrame(results_data)
        plot_calibration_results(results_df, eval_date, OUTPUT_DIR)
        
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()