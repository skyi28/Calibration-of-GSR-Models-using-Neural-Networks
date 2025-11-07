"""
Finds an optimal initial parameter guess for a 1+7 Hull-White model using a
parallelized Genetic Algorithm (GA) based on the Island Model.

Objective:
The primary goal of this script is to perform a robust, global search for the
optimal parameters of a Hull-White one-factor model with a single mean-reversion
parameter ('a') and a piecewise-constant volatility structure defined by seven
sigma parameters. Traditional gradient-based optimizers (like Levenberg-Marquardt)
are highly sensitive to their initial guess and can easily get trapped in local
minima. This script uses a GA to explore the vast parameter space and find a
high-quality set of initial parameters, which can then be used to seed a
traditional optimizer for fine-tuning.

Methodology:
The script implements a "Generalized Island Model" to parallelize the GA.
The total population of potential solutions (individuals) is split across
multiple "islands," with each island running an independent GA in a separate
process. This approach accelerates the search and enhances diversity.

Workflow:
1.  Initialization: For a specified market date, the script loads the zero-coupon
    yield curve and the swaption volatility cube. It initializes multiple
    populations (islands) with random parameter sets.
2.  Evolution: Within each island, the population evolves over a set number of
    generations. The fitness of each parameter set is its RMSE, calculated by
    pricing all relevant swaptions in QuantLib and comparing them to market data.
3.  Migration: At fixed intervals (epochs), the best individuals from each island
    migrate to a neighboring island in a ring topology, introducing new genetic
    material and preventing stagnation.
4.  Termination: After a total number of generations, the algorithm terminates.
5.  Output: The script identifies the best parameter set found across all islands
    and saves it to a NumPy file (`.npy`), ready for use in other calibration scripts.
"""

import datetime
import os
import pandas as pd
import numpy as np
import QuantLib as ql
from typing import List, Tuple
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

# --- HELPER FUNCTIONS ---
def parse_tenor(tenor_str: str) -> ql.Period:
    """
    Parses a tenor string (e.g., '10YR', '6MO') into a QuantLib Period object.

    Args:
        tenor_str (str): The tenor string to be parsed.

    Returns:
        ql.Period: The parsed tenor as a QuantLib Period.

    Raises:
        ValueError: If the parsing fails due to an invalid tenor string.
    """
    tenor_str = tenor_str.strip().upper()
    if 'YR' in tenor_str: num = int(tenor_str.replace('YR', '')); return ql.Period(num, ql.Years)
    if 'MO' in tenor_str: num = int(tenor_str.replace('MO', '')); return ql.Period(num, ql.Months)
    raise ValueError(f"Could not parse tenor string: {tenor_str}")

def parse_tenor_to_years(tenor_str: str) -> float:
    """
    Parses a tenor string (e.g., '10YR', '6MO') into a float representing years.

    Args:
        tenor_str (str): The tenor string to be parsed.

    Returns:
        float: The parsed tenor in years.

    Raises:
        ValueError: If the parsing fails due to an invalid tenor string.
    """
    tenor_str = tenor_str.strip().upper()
    if 'YR' in tenor_str: return float(int(tenor_str.replace('YR', '')))
    if 'MO' in tenor_str: return int(tenor_str.replace('MO', '')) / 12.0
    raise ValueError(f"Could not parse tenor string to years: {tenor_str}")

def _get_step_dates_from_expiries(
    ql_eval_date: ql.Date,
    included_expiries_yrs: List[float],
    num_segments: int
) -> List[ql.Date]:
    """
    Creates a list of step dates by dividing the included expiries into num_segments
    partitions. If there are not enough unique expiries, it reduces the number of
    segments and prints a warning message.

    Args:
        ql_eval_date (ql.Date): The evaluation date of the yield curve.
        included_expiries_yrs (List[float]): A list of unique expiries in years.
        num_segments (int): The number of segments to divide the expiries into.

    Returns:
        List[ql.Date]: A list of step dates, each representing a point in time where
            the yield curve is divided into a segment.
    """
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
    """
    Creates a QuantLib yield curve from a pandas DataFrame.

    Args:
        zero_curve_df (pd.DataFrame): A pandas DataFrame containing a time series of daily zero rates.
        eval_date (datetime.date): The date at which the yield curve should be evaluated.

    Returns:
        ql.RelinkableYieldTermStructureHandle: A QuantLib yield curve handle constructed from the input time series.
    """
    ql_eval_date = ql.Date(eval_date.day, eval_date.month, eval_date.year)
    dates = [ql_eval_date] + [ql.Date(d.day, d.month, d.year) for d in pd.to_datetime(zero_curve_df['Date'])]
    rates = [zero_curve_df['ZeroRate'].iloc[0]] + zero_curve_df['ZeroRate'].tolist()
    term_structure = ql.ZeroCurve(dates, rates, ql.Actual365Fixed(), ql.TARGET(), ql.Linear(), ql.Continuous, ql.Annual)
    term_structure.enableExtrapolation()
    handle = ql.RelinkableYieldTermStructureHandle()
    handle.linkTo(term_structure)
    return handle

def prepare_calibration_helpers(vol_cube_df: pd.DataFrame, term_structure_handle: ql.RelinkableYieldTermStructureHandle, min_expiry_years: float, min_tenor_years: float) -> List[ql.SwaptionHelper]:
    """
    Prepares a list of QuantLib SwaptionHelper objects from the volatility cube.

    Args:
        vol_cube_df (pd.DataFrame): A pandas DataFrame containing a volatility cube.
        term_structure_handle (ql.RelinkableYieldTermStructureHandle): A QuantLib yield curve handle.
        min_expiry_years (float): The minimum expiry in years for a swaption to be considered.
        min_tenor_years (float): The minimum tenor in years for a swaption to be considered.

    Returns:
        List[ql.SwaptionHelper]: A list of QuantLib SwaptionHelper objects representing the relevant swaptions.
    """
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
    """
    Listens to a multiprocessing queue and updates a tqdm progress bar
    accordingly. The progress bar is closed when a "STOP" message is received.

    Args:
        queue (multiprocessing.Queue): The queue to listen to.
        total (int): The total number of messages to expect.

    Returns:
        None
    """
    pbar = tqdm(total=total, desc="Epoch Generations")
    while True:
        message = queue.get()
        if message == "STOP":
            break
        pbar.update(1)
    pbar.close()

# --- ISLAND MODEL WORKER FUNCTION ---
def evolve_island(args: Tuple) -> Tuple[List[Tuple], Dict]:
    """
    Evolves a population of island models using a genetic algorithm.

    Args:
        args (Tuple): A tuple containing the island ID, population size, number of generations per epoch, genetic algorithm parameters, QuantLib initialization data, and a multiprocessing queue for progress updates.

    Returns:
        Tuple[List[Tuple], Dict]: A tuple containing the final population with its fitness values and a dictionary containing the fitness values of all evaluated individuals.

    Notes:
        - The genetic algorithm parameters are bounds, mutation rate, tournament size, elitism count, mutation strength, and the number of sigma segments.
        - The QuantLib initialization data consists of the evaluation date, zero curve DataFrame, volatility cube DataFrame, minimum expiry in years, and minimum tenor in years.
        - The fitness values are the root-mean-squared errors (RMSEs) of the implied volatilities versus the market volatilities in basis points.
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
    
    day_counter = ql.Actual365Fixed()
    included_expiries_yrs = sorted(list(set([
        day_counter.yearFraction(ql_eval_date, h.swaption().exercise().dates()[-1]) for h in helpers
    ])))
    
    sigma_step_dates = _get_step_dates_from_expiries(ql_eval_date, included_expiries_yrs, num_sigma_segments)

    fitness_cache = {}
    def calculate_rmse_local(params: Tuple) -> float:
        """
        Calculates the root-mean-squared error (RMSE) of the implied volatilities versus the market volatilities in basis points for a given set of parameters.

        Parameters
        ----------
        params : Tuple
            A tuple containing the a value and the sigma values to be used for the calculation.

        Returns
        -------
        float
            The root-mean-squared error of the implied volatilities versus the market volatilities in basis points.
        """

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

# --- MAIN WORKFLOW ---
if __name__ == '__main__':
    date_str = "03.08.2025"                   # Date for which we want to find the initial guess
    # --- GA and Model Hyperparameters ---
    NUM_SIGMA_SEGMENTS = 7
    NUM_ISLANDS = multiprocessing.cpu_count() # Set number of islands to the number of CPU cores, so that each core processes a different island
    MIGRATION_INTERVAL = 15                   # Number of generations between migrations
    MIGRATION_COUNT = 2                       # Number of individuals to migrate
    TOTAL_POPULATION = 512                    # Total number of individuals, should be divisible by NUM_ISLANDS so that each island has the same number of individuals
    TOTAL_GENERATIONS = 150                   # Total number of generations (TOTAL_GENERATIONS / MIGRATION_INTERVAL = number of epochs)
    MUTATION_RATE = 0.9                       # Controls the probability of mutation
    TOURNAMENT_SIZE = 3                       # Number of individuals participating in each tournament
    ELITISM_COUNT = 1                         # Number of individuals which have a garanteed place in the next generation
    INITIAL_MUTATION_STRENGTH = 1.0           # Initial mutation strength
    FINAL_MUTATION_STRENGTH = 0.05            # Final mutation strength since mutation strength is decayed over time
    
    POP_PER_ISLAND = max(10, TOTAL_POPULATION // NUM_ISLANDS)
    NUM_EPOCHS = TOTAL_GENERATIONS // MIGRATION_INTERVAL

    PARAM_BOUNDS = {'a': (0.005, 0.05), 'sigma': (0.00001, 0.001)} # Sigma bounds apply to all 7 sigmas
    MIN_EXPIRY_YEARS, MIN_TENOR_YEARS = 2.0, 2.0
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

        # --- MAIN LOOP WITH MANAGER ---
        with multiprocessing.Manager() as manager:
            progress_queue = manager.Queue()
            for epoch in range(NUM_EPOCHS):
                decay = (FINAL_MUTATION_STRENGTH / INITIAL_MUTATION_STRENGTH) ** (1/(NUM_EPOCHS-1)) if NUM_EPOCHS > 1 else 1.0
                mut_strength = INITIAL_MUTATION_STRENGTH * (decay ** epoch)
                ga_params = (PARAM_BOUNDS, MUTATION_RATE, TOURNAMENT_SIZE, ELITISM_COUNT, mut_strength, NUM_SIGMA_SEGMENTS)
                
                pool_args = [(i, island_populations[i], MIGRATION_INTERVAL, ga_params, ql_init_data, progress_queue) for i in range(NUM_ISLANDS)]

                print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} | Mutation Strength: {mut_strength:.4f} ---")
                
                total_generations_in_epoch = NUM_ISLANDS * MIGRATION_INTERVAL
                listener_thread = threading.Thread(target=progress_listener, args=(progress_queue, total_generations_in_epoch))
                listener_thread.start()
                
                with multiprocessing.Pool(NUM_ISLANDS) as pool:
                    results = pool.map(evolve_island, pool_args)

                progress_queue.put("STOP")
                listener_thread.join()

                # Process results and migrate
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

        # --- Final Evaluation ---
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
        print(results_df.to_string(index=False))
        
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()