#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import random
import argparse
from datetime import datetime
import logging

from solcx import compile_source
import solcx
from eth_utils import encode_hex, to_canonical_address
from z3 import Solver
from eth.db.account import Account

from evm import InstrumentedEVM
from detectors import DetectorExecutor
from engine import EvolutionaryFuzzingEngine
from engine.components.generator import Generator
from engine.components import Individual, Population
import config
from engine.analysis import SymbolicTaintAnalyzer
from engine.analysis import ExecutionTraceAnalyzer
from engine.environment import FuzzingEnvironment
from engine.operators import LinearRankingSelection
from engine.operators import DataDependencyLinearRankingSelection
from engine.operators import Crossover
from engine.operators import DataDependencyCrossover
from engine.operators import Mutation
from engine.fitness import fitness_function
from fuzzer.utils.transaction_seq_utils import check_cross_init, gen_trans, init_func

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from fuzzer.utils import settings
from provider import factory as provider_factory
from provider import context as provider_context
from provider.base import ProviderError
from utils.source_map import SourceMap
from utils.utils import initialize_logger, compile, get_interface_from_abi, get_pcs_and_jumpis, \
    get_function_signature_mapping
from utils.control_flow_graph import ControlFlowGraph

# ContraMaster Integration
import config
try:
    from fuzzer.analyzers.bookkeeping_identifier import BookkeepingVariableIdentifier
    contramaster_available = True
except ImportError:
    contramaster_available = False
    BookkeepingVariableIdentifier = None

try:
    from engine.analyzers.dataflow_analyzer import SmartContractAnalyzer
    from engine.components.rag_enhanced_generator import RAGEnhancedGenerator, create_rag_enhanced_generator
    from engine.components.rag_enhanced_population import RAGEnhancedPopulation
    from engine.components.llm_enhanced_generator import LLMEnhancedGenerator, create_llm_enhanced_generator
    rag_available = True
except ImportError as e:
    import logging
    logging.warning(f"Could not import RAG components: {e}")
    logging.warning("Will use standard generator if RAG is requested")
    rag_available = False

    class RAGEnhancedGenerator:
        pass

# LLM Evolution Integration
try:
    from engine.operators.llm_evolution import LLMClient, LLMMutation, LLMCrossover
    from engine.operators.llm_evolution.llm_controller import LLMController
    llm_evolution_available = True
except ImportError as e:
    import logging
    logging.warning(f"Could not import LLM evolution components: {e}")
    llm_evolution_available = False
    LLMClient = None
    LLMMutation = None
    LLMCrossover = None


class Fuzzer:
    def __init__(self, contract_name, abi, deployment_bytecode, runtime_bytecode, test_instrumented_evm,
                 blockchain_state, solver, args, seed, source_map=None, whole_compile_info=None):
        global logger

        logger = initialize_logger("Fuzzer  ")
        logger.title("Fuzzing contract %s", contract_name)

        cfg = ControlFlowGraph()
        cfg.build(runtime_bytecode, settings.EVM_VERSION)

        self.contract_name = contract_name
        self.interface, self.interface_mapper = get_interface_from_abi(abi)
        self.deployement_bytecode = deployment_bytecode
        self.blockchain_state = blockchain_state
        self.instrumented_evm = test_instrumented_evm
        self.solver = solver
        self.args = args
        self.depend_contracts = args.depend_contracts
        self.whole_compile_info = whole_compile_info

        self.overall_pcs, self.overall_jumpis = get_pcs_and_jumpis(runtime_bytecode)
        self.results = {"errors": {}}

        self.env = FuzzingEnvironment(
            instrumented_evm=self.instrumented_evm,
            contract_name=self.contract_name,
            solver=self.solver,
            results=self.results,
            symbolic_taint_analyzer=SymbolicTaintAnalyzer(),
            detector_executor=DetectorExecutor(source_map, get_function_signature_mapping(abi)),
            interface=self.interface,
            overall_pcs=self.overall_pcs,
            overall_jumpis=self.overall_jumpis,
            len_overall_pcs_with_children=0,
            other_contracts=list(),
            args=args,
            seed=seed,
            cfg=cfg,
            abi=abi
        )
        init_func(args.source)
        # Với bytecode mode (args.source = None), không cần check_cross_init
        if args.source is not None:
            assert check_cross_init(), "Cross-contract initialization failed"
            print("Cross-contract initialization successful...")
        else:
            print("Bytecode mode: Cross-contract initialization skipped")

    def deploy_depend_contracts(self, api_key=None, use_rag=True):
        generators = []
        if self.whole_compile_info is None:
            logger.error("No compilation info found, exiting program!")
            sys.exit(-1)

        if self.args.source and len(self.depend_contracts) != 0:
            for contract_name in self.depend_contracts:
                if contract_name == self.contract_name:
                    logger.error(f"{contract_name} is the same as the contract to be fuzzed!")
                    sys.exit(-1)

                contract = self.whole_compile_info[contract_name]
                if contract['abi'] and contract['evm']['bytecode']['object']:
                    interface, interface_mapper = get_interface_from_abi(contract['abi'])
                    deployement_bytecode = contract['evm']['bytecode']['object']

                    if "constructor" in interface:
                        del interface['constructor']

                    if "constructor" not in interface:
                        result = self.instrumented_evm.deploy_contract(
                            self.instrumented_evm.accounts[0],
                            deployement_bytecode
                        )

                        if result.is_error:
                            logger.error(
                                f"Problem deploying dependent contract {contract_name} using account "
                                f"{self.instrumented_evm.accounts[0]}. Error: {result._error}"
                            )
                        else:
                            contract_address = encode_hex(result.msg.storage_address)
                            self.instrumented_evm.accounts.append(contract_address)
                            self.env.nr_of_transactions += 1
                            logger.info(
                                f"Dependent contract {contract_name} deployed at {contract_address}, "
                                f"by {self.instrumented_evm.accounts[0]}"
                            )

                            settings.TRANS_INFO[contract_name] = contract_address
                            settings.DEPLOYED_CONTRACT_ADDRESS[contract_name] = contract_address

                            self.env.other_contracts.append(to_canonical_address(contract_address))
                            cc, _ = get_pcs_and_jumpis(
                                self.instrumented_evm.get_code(to_canonical_address(contract_address)).hex()
                            )
                            self.env.len_overall_pcs_with_children += len(cc)

                            # Use RAG generator for dependent contracts if available
                            if use_rag and rag_available and api_key:
                                try:
                                    generator = create_rag_enhanced_generator(
                                        interface=interface,
                                        bytecode=deployement_bytecode,
                                        accounts=self.instrumented_evm.accounts,
                                        contract=contract_address,
                                        api_key=api_key,
                                        analysis_result=None,  # No analysis for dependent contracts
                                        contract_name=contract_name,
                                        sol_path=self.args.source,
                                        interface_mapper=interface_mapper
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to create RAG generator for {contract_name}: {e}, using standard generator")
                                    generator = Generator(
                                        interface=interface,
                                        bytecode=deployement_bytecode,
                                        accounts=self.instrumented_evm.accounts,
                                        contract=contract_address,
                                        interface_mapper=interface_mapper,
                                        contract_name=contract_name,
                                        sol_path=self.args.source
                                    )
                            else:
                                generator = Generator(
                                    interface=interface,
                                    bytecode=deployement_bytecode,
                                    accounts=self.instrumented_evm.accounts,
                                    contract=contract_address,
                                    interface_mapper=interface_mapper,
                                    contract_name=contract_name,
                                    sol_path=self.args.source
                                )
                            generators.append(generator)

        return generators

    def run(self):
        settings.TRANS_INFO["contract_name"] = self.contract_name
        settings.TRANS_INFO["source_path"] = self.args.source
        settings.TRANS_INFO["start_time"] = str(datetime.now())
        settings.MAIN_CONTRACT_NAME = self.contract_name

        self.instrumented_evm.create_fake_accounts()

        # Get API key and RAG settings early for dependent contracts
        api_key = None
        if hasattr(self.args, 'api_key') and self.args.api_key:
            api_key = self.args.api_key
        elif config.get_google_api_key():
            api_key = config.get_google_api_key()
        
        use_rag = config.get_default_use_rag()
        if hasattr(self.args, 'use_rag'):
            use_rag = self.args.use_rag
        if hasattr(self.args, 'no_rag') and self.args.no_rag:
            use_rag = False

        # LLM settings shared by RAG and LLM Evolution
        openai_api_key = (getattr(self.args, 'openai_api_key', None) or
                         os.environ.get('OPENAI_API_KEY'))
        model = getattr(self.args, 'llm_model', None) or config.get_default_llm_model()
        provider_name = getattr(self.args, 'llm_provider', None) or config.get_default_llm_provider()
        ollama_endpoint = getattr(self.args, 'ollama_endpoint', None) or config.get_default_ollama_endpoint()
        disable_rag_llm = getattr(self.args, 'no_llm_rag', False)
        adaptive_llm_controller = None

        # Auto-detect provider from model name if not specified
        if (not provider_name) and model:
            if model.lower().startswith('gpt') or model.lower().startswith('o1'):
                provider_name = 'openai'
            elif model.lower().startswith('gemini'):
                provider_name = 'gemini'

        if self.args.cross_contract == 1:
            generators = self.deploy_depend_contracts(api_key=api_key, use_rag=use_rag)
        else:
            generators = []

        contract_address = None
        if self.args.source:
            for transaction in self.blockchain_state:
                if transaction['from'].lower() not in self.instrumented_evm.accounts:
                    self.instrumented_evm.accounts.append(
                        self.instrumented_evm.create_fake_account(transaction['from']))

                if not transaction['to']:
                    result = self.instrumented_evm.deploy_contract(
                        transaction['from'],
                        transaction['input'],
                        int(transaction['value']),
                        int(transaction['gas']),
                        int(transaction['gasPrice'])
                    )
                    if result.is_error:
                        logger.error(
                            f"Problem deploying contract {self.contract_name} using account "
                            f"{transaction['from']}. Error: {result._error}"
                        )
                        sys.exit(-2)
                    else:
                        contract_address = encode_hex(result.msg.storage_address)
                        self.instrumented_evm.accounts.append(contract_address)
                        self.env.nr_of_transactions += 1
                        logger.debug(f"Contract deployed at {contract_address}")
                        self.env.other_contracts.append(to_canonical_address(contract_address))
                        cc, _ = get_pcs_and_jumpis(
                            self.instrumented_evm.get_code(to_canonical_address(contract_address)).hex()
                        )
                        self.env.len_overall_pcs_with_children += len(cc)
                else:
                    input_data = {
                        "block": {},
                        "transaction": {
                            "from": transaction["from"],
                            "to": transaction["to"],
                            "gaslimit": int(transaction["gas"]),
                            "value": int(transaction["value"]),
                            "data": transaction["input"]
                        },
                        "global_state": {}
                    }
                    self.instrumented_evm.deploy_transaction(input_data, int(transaction["gasPrice"]))

            if "constructor" in self.interface:
                del self.interface["constructor"]

            if not contract_address and "constructor" not in self.interface:
                result = self.instrumented_evm.deploy_contract(
                    self.instrumented_evm.accounts[0],
                    self.deployement_bytecode,
                    deploy_args=self.args.constructor_args,
                    deploy_mode=settings.CROSS_INIT_MODE
                )

                if result.is_error:
                    logger.error(
                        f"Problem deploying contract {self.contract_name} using account "
                        f"{self.instrumented_evm.accounts[0]}. Error: {result._error}"
                    )
                    sys.exit(-2)
                else:
                    contract_address = encode_hex(result.msg.storage_address)
                    self.instrumented_evm.accounts.append(contract_address)
                    self.env.nr_of_transactions += 1
                    logger.info(f"Contract deployed at {contract_address}")
                    settings.TRANS_INFO[self.contract_name] = contract_address
                    settings.DEPLOYED_CONTRACT_ADDRESS[self.contract_name] = contract_address

            if contract_address in self.instrumented_evm.accounts:
                self.instrumented_evm.accounts.remove(contract_address)

            self.env.overall_pcs, self.env.overall_jumpis = get_pcs_and_jumpis(
                self.instrumented_evm.get_code(to_canonical_address(contract_address)).hex()
            )

        elif self.args.abi:
            contract_address = self.args.contract

        self.instrumented_evm.create_snapshot()

        # ContraMaster: Auto-identify bookkeeping variables
        bookkeeping_vars = []
        if config.is_contramaster_enabled() and config.ENABLE_BOOKKEEPING_AUTO_DETECT and contramaster_available:
            logger.info("ContraMaster: Identifying bookkeeping variables...")
            try:
                identifier = BookkeepingVariableIdentifier()
                bookkeeping_vars = identifier.identify(
                    contract_interface=self.env.abi,
                    contract_address=contract_address,
                    accounts=self.instrumented_evm.accounts,
                    web3=None  # No web3 needed for static analysis
                )
                if bookkeeping_vars:
                    logger.info(f"ContraMaster: Found bookkeeping variables: {bookkeeping_vars}")
                    # Set bookkeeping variable in detector
                    self.env.detector_executor.bookkeeping_variable = bookkeeping_vars[0]
                else:
                    logger.warning("ContraMaster: No bookkeeping variables identified")
            except Exception as e:
                logger.warning(f"ContraMaster bookkeeping identification error: {e}")

        # RAG is now default - always try to use RAG if available
        # Get API key from args, config, or environment
        api_key = None
        if hasattr(self.args, 'api_key') and self.args.api_key:
            api_key = self.args.api_key
        elif config.get_google_api_key():
            api_key = config.get_google_api_key()
        
        # Check if RAG should be disabled
        use_rag = config.get_default_use_rag()
        if hasattr(self.args, 'use_rag'):
            use_rag = self.args.use_rag
        if hasattr(self.args, 'no_rag') and self.args.no_rag:
            use_rag = False
        
        # Run dataflow analysis if RAG is enabled and source is available
        analysis_result = None
        if use_rag and api_key and self.args.source and not os.environ.get("SKIP_ANALYZER"):
            logger.info("Running dataflow analysis for RAG-enhanced fuzzing")
            try:
                analyzer = SmartContractAnalyzer(self.args.source, api_key, self.args.solc_path_cross)
                analysis_data = analyzer.analyze()
                analysis_result = analysis_data["analysis_result"]
                logger.info(
                    f"Dataflow analysis complete: found {len(analysis_result.get('critical_paths', []))} critical paths, "
                    f"{len(analysis_result.get('test_sequences', []))} test sequences, and "
                    f"{len(analysis_result.get('vulnerabilities', []))} potential vulnerabilities"
                )
            except Exception as e:
                logger.error(f"Error running dataflow analysis: {e}")
                logger.warning("Will continue without analysis result")
                analysis_result = None

        # Default to RAG-enhanced generator if available
        rag_generator = None
        if use_rag and rag_available:
            try:
                max_indiv_length = self.args.max_individual_length if hasattr(self.args, 'max_individual_length') else 10
                generator = create_rag_enhanced_generator(
                    interface=self.interface,
                    bytecode=self.deployement_bytecode,
                    accounts=self.instrumented_evm.accounts,
                    contract=contract_address,
                    api_key=api_key,  # Can be None, RAG will handle it
                    analysis_result=analysis_result,  # Can be None
                    contract_name=self.contract_name,
                    sol_path=self.args.source,
                    other_generators=generators,
                    interface_mapper=self.interface_mapper,
                    max_individual_length=max_indiv_length,
                    llm_model=model,
                    llm_provider=provider_name,
                    openai_api_key=openai_api_key,
                    disable_rag_llm=disable_rag_llm,
                    adaptive_llm_controller=adaptive_llm_controller,
                )
                logger.info("Using RAG-enhanced generator (default)")
                rag_generator = generator
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                logger.error(f"Error creating RAG-enhanced generator: {e}\n{tb}")
                logger.warning("Falling back to standard generator")
                generator = Generator(
                    interface=self.interface,
                    bytecode=self.deployement_bytecode,
                    accounts=self.instrumented_evm.accounts,
                    contract=contract_address,
                    other_generators=generators,
                    interface_mapper=self.interface_mapper,
                    contract_name=self.contract_name,
                    sol_path=self.args.source
                )
        else:
            # Fallback to standard generator if RAG is disabled or not available
            if not rag_available:
                logger.info("RAG components not available, using standard generator")
            elif not use_rag:
                logger.info("RAG disabled, using standard generator")
            generator = Generator(
                interface=self.interface,
                bytecode=self.deployement_bytecode,
                accounts=self.instrumented_evm.accounts,
                contract=contract_address,
                other_generators=generators,
                interface_mapper=self.interface_mapper,
                contract_name=self.contract_name,
                sol_path=self.args.source
            )

        all_generators = [generator] + generators
        for gen in generators:
            gen.update_other_generators(all_generators, generator.total_interface_mapper)

        # Kích thước population mặc định: 2 * số hàm, nhưng luôn tối thiểu 10 cá thể
        # để đảm bảo độ đa dạng, đặc biệt với các contract rất nhỏ (ví dụ chỉ 1-2 hàm).
        size = max(10, 2 * len(self.interface))
        if hasattr(self.args, 'population_size') and self.args.population_size:
            size = self.args.population_size

        # Default to RAG-enhanced population if using RAG generator
        if isinstance(generator, RAGEnhancedGenerator) and rag_available:
            logger.info("Using RAG-enhanced population (default)")
            try:
                max_indiv_length = self.args.max_individual_length if hasattr(self.args, 'max_individual_length') else 10
                population = RAGEnhancedPopulation(
                    indv_template=Individual(generator=generator, other_generators=generators),
                    indv_generator=generator,
                    size=size,
                    other_generators=generators
                ).init(init_seed=False)

                if hasattr(population, 'max_log_individuals'):
                    population.max_log_individuals = min(5, size)
                if hasattr(population, 'max_individual_length'):
                    population.max_individual_length = max_indiv_length
            except Exception as e:
                logger.error(f"Error creating RAG-enhanced population: {e}")
                logger.warning("Falling back to standard population")
                population = Population(
                    indv_template=Individual(generator=generator, other_generators=generators),
                    indv_generator=generator,
                    size=size,
                    other_generators=generators
                ).init(init_seed=False)
        else:
            # Use standard population if not using RAG generator
            population = Population(
                indv_template=Individual(generator=generator, other_generators=generators),
                indv_generator=generator,
                size=size,
                other_generators=generators
            ).init(init_seed=False)

        if self.args.data_dependency:
            selection = DataDependencyLinearRankingSelection(env=self.env)
            base_crossover = DataDependencyCrossover(pc=settings.PROBABILITY_CROSSOVER, env=self.env)
            base_mutation = Mutation(pm=settings.PROBABILITY_MUTATION)
        else:
            selection = LinearRankingSelection()
            base_crossover = Crossover(pc=settings.PROBABILITY_CROSSOVER)
            base_mutation = Mutation(pm=settings.PROBABILITY_MUTATION)

        # LLM Evolution Integration (reuse settings computed above)
        api_key = (getattr(self.args, 'api_key', None) or
                   os.environ.get('GOOGLE_API_KEY') or
                   os.environ.get('GEMINI_API_KEY'))

        if llm_evolution_available and (api_key or provider_name == "ollama" or openai_api_key or provider_name == "openai"):
            try:
                llm_client = LLMClient.from_config(
                    model=model,
                    provider_name=provider_name,
                    api_key=api_key,
                    ollama_endpoint=ollama_endpoint,
                    openai_api_key=openai_api_key,
                    temperature=0.8
                )

                max_generations = getattr(settings, 'MAX_GENERATIONS', 1000)
                adaptive_llm_controller = LLMController(
                    max_generations=max_generations,
                    exploration_ratio=0.2,
                    transition_ratio=0.3,
                )

                mutation = LLMMutation(
                    llm_client=llm_client,
                    mutation_probability=0.5,
                    adaptive_llm_controller=adaptive_llm_controller,
                )

                crossover = LLMCrossover(
                    llm_client=llm_client,
                    crossover_probability=0.3,
                    adaptive_llm_controller=adaptive_llm_controller,
                )

                logger.info(f"LLM evolution enabled: model={model}, provider={provider_name}")
                print(f"[LLM-EVO] Enabled mutation/crossover with model={model}, provider={provider_name}", flush=True)
            except Exception as e:
                logger.warning(f"Failed to initialize LLM evolution: {e}")
                logger.warning("Falling back to traditional operators")
                crossover = base_crossover
                mutation = base_mutation
                adaptive_llm_controller = None
        else:
            if not llm_evolution_available:
                logger.info("LLM evolution NOT available - using standard operators")
            elif not api_key and provider_name != "ollama" and not openai_api_key and provider_name != "openai":
                logger.info("LLM evolution requires API key (Gemini/OpenAI) or Ollama - using standard operators")
            else:
                logger.info("LLM evolution DISABLED - using standard operators")
            crossover = base_crossover
            mutation = base_mutation
            adaptive_llm_controller = None

        engine = EvolutionaryFuzzingEngine(
            population=population,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            mapping=get_function_signature_mapping(self.env.abi),
            adaptive_llm_controller=adaptive_llm_controller,
        )
        settings.CURRENT_ENGINE = engine
        engine.fitness_register(lambda x: fitness_function(x, self.env))
        engine.analysis.append(ExecutionTraceAnalyzer(self.env))

        self.env.execution_begin = time.time()
        self.env.population = population
        settings.GLOBAL_ENV = self.env

        # Respect CLI generations override; fall back to settings.GENERATIONS/MAX_GENERATIONS
        max_generations = getattr(settings, 'MAX_GENERATIONS', None)
        if not max_generations:
            max_generations = getattr(settings, 'GENERATIONS', 1000)
        # Run with plateau detection - can still stop early if enabled
        plateau_gens = getattr(settings, 'PLATEAU_GENERATIONS', 10)
        plateau_threshold = getattr(settings, 'PLATEAU_THRESHOLD', 0.1)
        stop_on_plateau = False if getattr(self.args, "no_plateau", False) else getattr(settings, 'STOP_ON_PLATEAU', True)

        print(f"\n🚀 Starting fuzzing (max_generations={max_generations}, stop_on_plateau={stop_on_plateau})", flush=True)
        print(f"   Will stop if no coverage improvement for {plateau_gens} generations\n", flush=True)

        engine.run(
            ng=max_generations,
            stop_on_plateau=stop_on_plateau,
            plateau_generations=plateau_gens,
            plateau_threshold=plateau_threshold
        )

        execution_time = time.time() - self.env.execution_begin

        # Count unique branches visited (each JUMPI has 2 possible outcomes: 0 and 1)
        # Total branches = number of JUMPI instructions * 2 (each has true/false path)
        total_branches = len(self.env.overall_jumpis) * 2
        branch_coverage = 0
        for pc in self.env.visited_branches:
            branch_coverage += len(self.env.visited_branches[pc])
        branch_coverage_percentage = 0
        if total_branches > 0:
            # Ensure branch coverage doesn't exceed 100%
            branch_coverage = min(branch_coverage, total_branches)
            branch_coverage_percentage = (branch_coverage / total_branches) * 100
        self.env.branch_coverage = branch_coverage_percentage

        detailed_results = {
            "code_coverage": len(self.env.code_coverage) / len(self.env.overall_pcs) * 100 if len(self.env.overall_pcs) > 0 else 0,
            "branch_coverage": branch_coverage_percentage,
            "total_transactions": self.env.nr_of_transactions,
            "unique_transactions": len(self.env.unique_individuals),
            "execution_time": execution_time,
            "memory_usage": self.env.memory_consumption if hasattr(self.env, "memory_consumption") else 0,
            "errors": self.results["errors"],
        }

        if rag_generator:
            try:
                output_dir = "../output/fuzzing_results"
                if self.args.results:
                    output_dir = os.path.dirname(self.args.results) or "."

                rag_generator.finalize_fuzzing(detailed_results, output_dir)
                logger.info(f"RAG-enhanced fuzzing results saved to {output_dir}")
            except Exception as e:
                logger.error(f"Error finalizing RAG-enhanced fuzzing: {e}")

        if self.env.args.cfg:
            if self.env.args.source:
                self.env.cfg.save_control_flow_graph(
                    os.path.splitext(self.env.args.source)[0] + '-' + self.contract_name, 'pdf'
                )
            elif self.env.args.abi:
                self.env.cfg.save_control_flow_graph(
                    os.path.join(os.path.dirname(self.env.args.abi), self.contract_name), 'pdf'
                )

        self.instrumented_evm.reset()
        settings.TRANS_INFO["end_time"] = str(datetime.now())


def main():
    args = launch_argument_parser()

    logger = initialize_logger("Main    ")

    if args.results and os.path.exists(args.results):
        os.remove(args.results)
        logger.info(f"Contract {args.source} has already been analyzed: {args.results}")
        logger.info(f"Original test output file {args.results} has been deleted")

    if args.seed:
        seed = args.seed
        if "PYTHONHASHSEED" not in os.environ:
            logger.debug("Please set PYTHONHASHSEED to '1' for Python's hash function to behave deterministically.")
    else:
        seed = random.random()
    random.seed(seed)
    logger.title("Initializing seed to %s", seed)

    instrumented_evm = InstrumentedEVM(settings.RPC_HOST, settings.RPC_PORT)
    instrumented_evm.set_vm_by_name(settings.EVM_VERSION)

    solver = Solver()
    solver.set("timeout", settings.SOLVER_TIMEOUT)

    blockchain_state = []
    if args.blockchain_state:
        if args.blockchain_state.endswith(".json"):
            with open(args.blockchain_state) as json_file:
                for line in json_file.readlines():
                    blockchain_state.append(json.loads(line))
        elif args.blockchain_state.isnumeric():
            settings.BLOCK_HEIGHT = int(args.blockchain_state)
            instrumented_evm.set_vm(settings.BLOCK_HEIGHT)
        else:
            logger.error(f"Unsupported input file: {args.blockchain_state}")
            sys.exit(-1)

    if args.source:
        if args.source.endswith(".sol"):
            compiler_output = compile(args.solc_version, settings.EVM_VERSION, args.source)
            if not compiler_output:
                logger.error(f"No compiler output for: {args.source}")
                sys.exit(-1)
            for contract_name, contract in compiler_output['contracts'][args.source].items():
                if args.contract and contract_name != args.contract:
                    continue
                if contract['abi'] and contract['evm']['bytecode']['object'] and contract['evm']['deployedBytecode']['object']:
                    source_map = SourceMap(':'.join([args.source, contract_name]), compiler_output)
                    Fuzzer(
                        contract_name,
                        contract["abi"],
                        contract['evm']['bytecode']['object'],
                        contract['evm']['deployedBytecode']['object'],
                        instrumented_evm,
                        blockchain_state,
                        solver,
                        args,
                        seed,
                        source_map,
                        compiler_output['contracts'][args.source]
                    ).run()
        else:
            logger.error(f"Unsupported input file: {args.source}")
            sys.exit(-1)

    if args.abi:
        with open(args.abi) as json_file:
            abi = json.load(json_file)
            runtime_bytecode = instrumented_evm.get_code(to_canonical_address(args.contract)).hex()
            Fuzzer(args.contract, abi, None, runtime_bytecode, instrumented_evm, blockchain_state, solver, args, seed).run()


def launch_argument_parser():
    parser = argparse.ArgumentParser()

    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("-s", "--source", type=str, help="Solidity smart contract source code file (.sol).")
    group1.add_argument("-a", "--abi", type=str, help="Smart contract ABI file (.json).")

    parser.add_argument(
        "-c", "--contract", type=str,
        help="Contract name to be fuzzed (if Solidity source code file provided) or blockchain contract address (if ABI file provided)."
    )
    parser.add_argument(
        "-b", "--blockchain-state", type=str,
        help="Initialize fuzzer with a blockchain state by providing a JSON file (if Solidity source code file provided) or a block number (if ABI file provided)."
    )

    parser.add_argument(
        "--solc",
        help=f"Solidity compiler version (default '{solcx.get_solc_version()}'). Installed compiler versions: {solcx.get_installed_solc_versions()}.",
        action="store", dest="solc_version", type=str
    )
    parser.add_argument(
        "--evm",
        help=f"Ethereum VM (default '{settings.EVM_VERSION}'). Available VMs: 'homestead', 'byzantium', or 'petersburg'.",
        action="store", dest="evm_version", type=str
    )

    group3 = parser.add_mutually_exclusive_group(required=False)
    group3.add_argument(
        "-g", "--generations", help=f"Number of generations (default {settings.GENERATIONS}).",
        action="store", dest="generations", type=int
    )
    group3.add_argument(
        "-t", "--timeout", help="Number of seconds for fuzzer to stop.",
        action="store", dest="global_timeout", type=int
    )
    parser.add_argument(
        "-n", "--population-size", help="Size of the population.",
        action="store", dest="population_size", type=int
    )
    parser.add_argument(
        "-pc", "--probability-crossover", help="Probability of crossover.",
        action="store", dest="probability_crossover", type=float
    )
    parser.add_argument(
        "-pm", "--probability-mutation", help="Probability of mutation.",
        action="store", dest="probability_mutation", type=float
    )

    parser.add_argument("-r", "--results", type=str, help="Folder or JSON file where results should be stored.")
    parser.add_argument("--seed", type=float, help="Initialize the random number generator with a given seed.")
    parser.add_argument("--cfg", help="Build control-flow graph and highlight code coverage.", action="store_true")
    parser.add_argument("--rpc-host", help="Ethereum client RPC hostname.", action="store", dest="rpc_host", type=str)
    parser.add_argument("--rpc-port", help="Ethereum client RPC port.", action="store", dest="rpc_port", type=int)

    parser.add_argument(
        "--api-key", help="Google API Key for LLM-enhanced fuzzing.", action="store", dest="api_key", type=str
    )
    parser.add_argument(
        "--openai-api-key", help="OpenAI API Key for LLM-enhanced fuzzing.", action="store", dest="openai_api_key", type=str
    )
    parser.add_argument(
        "--model", help="LLM model identifier.", action="store", dest="llm_model", type=str
    )
    parser.add_argument(
        "--provider", help="LLM provider (auto | gemini | ollama | openai).", action="store", dest="llm_provider", type=str
    )
    parser.add_argument(
        "--no-llm-rag", help="Disable LLM inside RAG (use FAISS similarity only).", action="store_true", dest="no_llm_rag"
    )
    parser.add_argument(
        "--ollama-endpoint", help="Ollama endpoint URL.", action="store", dest="ollama_endpoint", type=str
    )
    parser.add_argument(
        "--ollama-thinking", help="Enable thinking trace for Ollama models.", action="store", dest="ollama_thinking", type=str
    )
    parser.add_argument(
        "--use-llm", help="Enable LLM-enhanced fuzzing (requires --api-key).", action="store_true", dest="use_llm"
    )
    parser.add_argument(
        "--use-rag", help="Enable RAG-enhanced fuzzing with dataflow analysis (default: enabled).",
        action="store_true", dest="use_rag"
    )
    parser.add_argument(
        "--no-rag", help="Disable RAG-enhanced fuzzing (use standard generator).",
        action="store_true", dest="no_rag"
    )
    parser.add_argument(
        "--audit-file", help="Path to audit report for context-aware LLM fuzzing.",
        action="store", dest="audit_file", type=str
    )
    parser.add_argument(
        "--skip-dataflow-analysis", help="Skip dataflow analysis even when using RAG.",
        action="store_true", dest="skip_dataflow"
    )


    parser.add_argument(
        "--data-dependency", help="Disable/Enable data dependency analysis: 0 - Disable, 1 - Enable (default: 1)",
        action="store", dest="data_dependency", type=int, default=1
    )
    parser.add_argument(
        "--constraint-solving", help="Disable/Enable constraint solving: 0 - Disable, 1 - Enable (default: 1)",
        action="store", dest="constraint_solving", type=int
    )
    parser.add_argument(
        "--environmental-instrumentation",
        help="Disable/Enable environmental instrumentation: 0 - Disable, 1 - Enable (default: 1)",
        action="store", dest="environmental_instrumentation", type=int
    )
    parser.add_argument(
        "--max-individual-length", help=f"Maximal length of an individual (default: {settings.MAX_INDIVIDUAL_LENGTH})",
        action="store", dest="max_individual_length", type=int
    )
    parser.add_argument(
        "--no-plateau", help="Disable plateau early-stop", action="store_true", dest="no_plateau"
    )
    parser.add_argument(
        "--plateau-generations", help="Generations without improvement before stop (default: 10)",
        action="store", dest="plateau_generations", type=int
    )
    parser.add_argument(
        "--plateau-threshold", help="Coverage improvement threshold in %% (default: 0.1)",
        action="store", dest="plateau_threshold", type=float
    )
    parser.add_argument(
        "--max-symbolic-execution",
        help=f"Maximum number of symbolic execution calls before resetting population (default: {settings.MAX_SYMBOLIC_EXECUTION})",
        action="store", dest="max_symbolic_execution", type=int
    )

    parser.add_argument(
        "--cross-contract", type=int, help="Open cross contract mode, open -- 1, close -- 2 (default)",
        action="store", dest="cross_contract", default=2
    )
    parser.add_argument(
        "--depend-contracts", type=str, nargs="*",
        help="Main fuzzed contract depends on these contracts, you should give some names.",
        dest="depend_contracts"
    )
    parser.add_argument(
        "--trans-json-path", type=str, help="Location to save transaction info to JSON",
        dest="trans_json_path"
    )
    parser.add_argument(
        "--solc-path-cross", type=str, help="Solc path, used by cross-slither", dest="solc_path_cross"
    )
    parser.add_argument(
        "--constructor-args", type=str, nargs="*",
        help="Constructor args, like: [address, uint, ...]", dest="constructor_args"
    )
    parser.add_argument(
        "--open-trans-comp", type=int, help="Open cross transaction mode, open -- 1 (default), close -- 2",
        action="store", dest="trans_comp", default=1
    )
    parser.add_argument(
        "--trans-mode", type=int, help="Transaction support mode, open other -- 1, no exec other -- 2",
        default=1, dest="trans_mode"
    )
    parser.add_argument(
        "--p-open-cross", type=int, help="Use cross transaction probability: (1~8)", default=5, dest="p_open_cross"
    )
    parser.add_argument(
        "--cross-init-mode", type=int, help="Cross init mode: 1 -- specify, 2 -- random, 3 -- close",
        default=1, dest="cross_init_mode"
    )
    parser.add_argument(
        "--duplication", type=str, help="Duplication mode: 0 -- close, 1 -- open", default='0', dest="duplication"
    )

    version = "ConFuzzius - Version 0.0.2 - "
    version += "\"By three methods we may learn wisdom:\n"
    version += "First, by reflection, which is noblest;\n"
    version += "Second, by imitation, which is easiest;\n"
    version += "And third by experience, which is the bitterest.\"\n"
    parser.add_argument("-v", "--version", action="version", version=version)

    args = parser.parse_args()

    if not args.llm_model:
        args.llm_model = config.get_default_llm_model()
    if not args.llm_provider:
        args.llm_provider = config.get_default_llm_provider()
    if not args.ollama_endpoint:
        args.ollama_endpoint = config.get_default_ollama_endpoint()
    if args.ollama_thinking is None:
        args.ollama_thinking = config.get_default_ollama_thinking()

    if not args.contract:
        args.contract = ""

    if args.source and args.contract.startswith("0x"):
        parser.error("--source requires --contract to be a name, not an address.")
    if args.source and args.blockchain_state and args.blockchain_state.isnumeric():
        parser.error("--source requires --blockchain-state to be a file, not a number.")

    if args.abi and not args.contract.startswith("0x"):
        parser.error("--abi requires --contract to be an address, not a name.")
    if args.abi and args.blockchain_state and not args.blockchain_state.isnumeric():
        parser.error("--abi requires --blockchain-state to be a number, not a file.")

    if args.use_rag and args.use_llm:
        parser.error("Cannot use both --use-rag and --use-llm together. Choose one method.")
    if args.use_llm and not args.api_key:
        parser.error("LLM enhanced fuzzing requires --api-key parameter.")
    if args.use_rag and not args.api_key:
        logging.warning("RAG enabled without API key - will use fallback mode")

    if args.evm_version:
        settings.EVM_VERSION = args.evm_version
    if not args.solc_version:
        args.solc_version = solcx.get_solc_version()
    if args.generations:
        settings.GENERATIONS = args.generations
        settings.MAX_GENERATIONS = args.generations
    if args.global_timeout:
        settings.GLOBAL_TIMEOUT = args.global_timeout
    if args.population_size:
        settings.POPULATION_SIZE = args.population_size
    if args.probability_crossover:
        settings.PROBABILITY_CROSSOVER = args.probability_crossover
    if args.probability_mutation:
        settings.PROBABILITY_MUTATION = args.probability_mutation

    if args.data_dependency is None:
        args.data_dependency = 1
    if args.constraint_solving is None:
        args.constraint_solving = 1
    if args.environmental_instrumentation is None:
        args.environmental_instrumentation = 1

    if args.environmental_instrumentation == 1:
        settings.ENVIRONMENTAL_INSTRUMENTATION = True
    elif args.environmental_instrumentation == 0:
        settings.ENVIRONMENTAL_INSTRUMENTATION = False

    if args.max_individual_length:
        settings.MAX_INDIVIDUAL_LENGTH = args.max_individual_length
    if args.max_symbolic_execution:
        settings.MAX_SYMBOLIC_EXECUTION = args.max_symbolic_execution
    if args.no_plateau:
        settings.STOP_ON_PLATEAU = False
    if args.plateau_generations:
        settings.PLATEAU_GENERATIONS = args.plateau_generations
    if args.plateau_threshold:
        settings.PLATEAU_THRESHOLD = args.plateau_threshold

    provider_logger = logging.getLogger("LLMProvider")
    provider_context.clear_provider()
    try:
        openai_api_key = (getattr(args, 'openai_api_key', None) or
                         os.environ.get('OPENAI_API_KEY'))
        # Chỉ truyền tham số Ollama khi provider được yêu cầu là Ollama
        ollama_kwargs = {}
        if (args.llm_provider or "").lower() == "ollama":
            ollama_kwargs = {
                "ollama_endpoint": args.ollama_endpoint,
                "ollama_thinking": args.ollama_thinking,
            }

        provider = provider_factory.create_provider(
            model=args.llm_model,
            provider_name=args.llm_provider,
            api_key=args.api_key,
            openai_api_key=openai_api_key,
            default_provider=config.get_default_llm_provider(),
            **ollama_kwargs,
        )
        provider_context.set_provider(provider)
        provider_logger.info(
            "Initialized LLM provider '%s' with model '%s'",
            provider.__class__.__name__,
            args.llm_model,
        )
    except ProviderError as exc:
        provider_logger.warning("Failed to initialize LLM provider: %s", exc)
        provider_context.clear_provider()
    except Exception as exc:  # pragma: no cover - unexpected failures
        provider_logger.error("Unexpected error during provider initialization: %s", exc)
        provider_context.clear_provider()

    if args.abi:
        settings.REMOTE_FUZZING = True

    if args.rpc_host:
        settings.RPC_HOST = args.rpc_host
    if args.rpc_port:
        settings.RPC_PORT = args.rpc_port

    if args.contract is None or args.contract == "" or args.cross_contract == 2:
        args.cross_contract = 2
        args.depend_contracts = []
        args.trans_json_path = None
    else:
        if args.contract is None or args.contract == "":
            print('\033[42;31m!!!!!!If open cross contract mode, you need specify a main contract which will be fuzzed!!!!!!\033[0m')
            print('\033[42;31m!!!!!!Use --contract [Example]!!!!!!\033[0m')
            sys.exit(-1)
        if args.depend_contracts is None:
            print('\033[42;31m!!!!!!If open cross contract mode, you need specify some contract names which depended by main contract!!!!!!\033[0m')
            print('\033[42;31m!!!!!!Use --depend-contracts [A B C]!!!!!!\033[0m')
            sys.exit(-1)
        if args.constructor_args is None:
            print('\033[42;31m!!!!!!If open cross contract mode, you need specify some constructor args!!!!!!\033[0m')
            print('\033[42;31m!!!!!!Use --constructor-args [address, uint, ...]!!!!!!\033[0m')
            sys.exit(-1)

    if args.trans_json_path is not None:
        settings.TRANS_INFO_JSON_PATH = args.trans_json_path
        print(f'\033[42;31m!!!!!!Set JSON path for transaction info: {settings.TRANS_INFO_JSON_PATH}!!!!!!\033[0m')
        if os.path.exists(settings.TRANS_INFO_JSON_PATH):
            print(f'\033[42;31m!!!!!!JSON path for transaction info {settings.TRANS_INFO_JSON_PATH} already exists, overwritten!!!!!!\033[0m')

    if args.trans_comp == 1:
        settings.TRANS_COMP_OPEN = True
    elif args.trans_comp == 2:
        settings.TRANS_COMP_OPEN = False

    settings.MAIN_CONTRACT_NAME = args.contract
    settings.SOLC_PATH_CROSS = args.solc_path_cross
    settings.P_OPEN_CROSS = args.p_open_cross
    settings.CROSS_INIT_MODE = args.cross_init_mode
    settings.TRANS_SUPPORT_MODE = args.trans_mode

    if args.duplication == '0':
        settings.DUPLICATION = True
    else:
        settings.DUPLICATION = False

    if args.cross_contract == 1 and settings.SOLC_PATH_CROSS is None:
        print('\033[42;31m!!!!!!You need to specify a solc path!!!!!!\033[0m')
        sys.exit(-1)

    return args


if __name__ == '__main__':
    main()