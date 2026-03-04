from config import *
from queue import Queue
from slither import Slither
from slither.core.declarations import Contract
from typing import Tuple, List
from slither.core.expressions import TypeConversion, Identifier, AssignmentOperation
from slither.core.solidity_types import UserDefinedType

logger = get_logger()


@logger.catch()
def analysis_depend_contract(
        file_path: str,
        _contract_name: str,
        _solc_version: str,
        _solc_path
) -> Tuple[List, Slither]:
    res = set()
    sl = Slither(file_path, solc=_solc_path)

    to_be_deep_analysis = Queue()
    to_be_deep_analysis.put(_contract_name)

    while not to_be_deep_analysis.empty():
        c = to_be_deep_analysis.get()
        contract = sl.get_contract_from_name(c)

        if len(contract) != 1:
            logger.warning("Theoretically, only one contract should be found by name")
            return [], sl

        contract = contract[0]

        for v in contract.all_state_variables_written:
            if (
                not v.initialized
                and isinstance(v.type, UserDefinedType)
                and hasattr(v.type, "type")
                and isinstance(v.type.type, Contract)
            ):
                res.add(v.type.type.name)
                logger.debug(
                    f"Dependent contract detected via written state variable: {v.type.type.name}"
                )

        for f in contract.functions:
            for p in f.parameters:
                if (
                    isinstance(p.type, UserDefinedType)
                    and hasattr(p.type, "type")
                    and isinstance(p.type.type, Contract)
                ):
                    res.add(p.type.type.name)
                    logger.debug(
                        f"Dependent contract detected via function parameter: {p.type.type.name}"
                    )

            for v in f.variables_written:
                if (
                    hasattr(v, "type")
                    and isinstance(v.type, UserDefinedType)
                    and hasattr(v.type, "type")
                    and isinstance(v.type.type, Contract)
                ):
                    res.add(v.type.type.name)
                    logger.debug(
                        f"Dependent contract detected via written variable: {v.type.type.name}"
                    )

        for inherit in contract.inheritance:
            if inherit.name not in res:
                to_be_deep_analysis.put(inherit.name)

    if _contract_name in res:
        logger.debug("Main contract found in dependency list, removing it")
        res.remove(_contract_name)

    try:
        compilation_unit = sl.compilation_units[0].crytic_compile_compilation_unit
    except Exception as exc:
        logger.debug(f"Failed to obtain compilation unit: {exc}")
        compilation_unit = None

    def has_bytecode(contract_name: str) -> bool:
        if compilation_unit is None:
            return False
        try:
            bytecode_rt = getattr(compilation_unit, "bytecode_runtime", None)
            if callable(bytecode_rt):
                rt = bytecode_rt(contract_name)
                return rt not in ("", None)
        except Exception:
            pass
        return False

    if compilation_unit is not None:
        for depend_c in res.copy():
            if not has_bytecode(depend_c):
                logger.debug(
                    f"Dependent contract {depend_c} has empty bytecode, removed"
                )
                res.remove(depend_c)

    interface_contracts = []
    implementable_contracts = []

    try:
        for contract in sl.contracts:
            contract_name = contract.name

            if contract_name in res:
                continue

            has_rt_bytecode = has_bytecode(contract_name)

            has_implemented_functions = False
            if contract.functions:
                has_implemented_functions = any(
                    f.is_implemented for f in contract.functions if not f.is_constructor
                )

            if not has_rt_bytecode and not has_implemented_functions:
                interface_contracts.append(contract_name)
            else:
                implementable_contracts.append(contract_name)

    except Exception as exc:
        logger.debug(f"Interface detection failed: {exc}")

    depend_list = list(res)
    total_contracts = len(sl.contracts)
    depend_count = len(res)

    logger.info(
        f"Dependent contracts: {depend_list}, "
        f"Total contracts: {total_contracts}, "
        f"To deploy: {depend_count}"
    )

    if interface_contracts:
        logger.info(
            f"⚠️  Detected {len(interface_contracts)} interface contracts (not fuzzable): "
            f"{interface_contracts}"
        )

    if implementable_contracts:
        logger.info(
            f"✅ Fuzzable contracts (with implementation): {implementable_contracts}"
        )

    return list(res), sl


def get_implementable_contracts(file_path: str, solc_path: str) -> List[str]:
    """
    Get list of implementable (fuzzable) contracts from a Solidity file.
    
    Returns contracts that have bytecode or implemented functions,
    excluding interface contracts.
    """
    try:
        sl = Slither(file_path, solc=solc_path)
    except Exception as exc:
        logger.debug(f"Failed to analyze {file_path} with Slither: {exc}")
        return []
    
    try:
        compilation_unit = sl.compilation_units[0].crytic_compile_compilation_unit
    except Exception as exc:
        logger.debug(f"Failed to obtain compilation unit: {exc}")
        compilation_unit = None
    
    def has_bytecode(contract_name: str) -> bool:
        if compilation_unit is None:
            return False
        try:
            contract_obj = compilation_unit.contracts_by_name.get(contract_name)
            if contract_obj and contract_obj.bytecode_runtime:
                return contract_obj.bytecode_runtime not in ("", None)
        except Exception:
            pass
        return False
    
    implementable_contracts = []
    
    try:
        for contract in sl.contracts:
            contract_name = contract.name
            
            # Skip libraries
            if contract.is_library:
                logger.debug(f"Skipping library: {contract_name}")
                continue
            
            has_rt_bytecode = has_bytecode(contract_name)
            has_implemented_functions = False
            
            if contract.functions:
                has_implemented_functions = any(
                    f.is_implemented for f in contract.functions if not f.is_constructor
                )
            
            # Check if contract is abstract (has unimplemented public/external functions)
            # but still allow if it has bytecode (might be deployed via inheritance)
            is_abstract = False
            if contract.functions and not has_rt_bytecode:
                unimplemented_public = [
                    f for f in contract.functions 
                    if not f.is_constructor 
                    and not f.is_implemented 
                    and f.visibility in ["public", "external"]
                ]
                if unimplemented_public:
                    is_abstract = True
                    logger.debug(f"Skipping abstract contract: {contract_name} (has {len(unimplemented_public)} unimplemented public/external functions)")
            
            # Include contract if:
            # 1. It has runtime bytecode (can be deployed), OR
            # 2. It has implemented functions AND is not abstract
            if has_rt_bytecode or (has_implemented_functions and not is_abstract):
                implementable_contracts.append(contract_name)
    except Exception as exc:
        logger.debug(f"Failed to detect implementable contracts: {exc}")
    
    return implementable_contracts


def analysis_main_contract_constructor(
        file_path: str,
        _contract_name: str,
        sl: Slither = None
):
    if sl is None:
        sl = Slither(file_path, solc=SOLC_BIN_PATH)

    contract = sl.get_contract_from_name(_contract_name)
    assert len(contract) == 1, "Theoretically, only one contract should be found by name"

    contract = contract[0]
    logger.info(f"=== Start constructor analysis for {_contract_name} ===")

    constructor = contract.constructor
    if constructor is None:
        logger.info("Contract has no constructor")
        return []

    logger.info("Constructor found, analyzing parameters")
    res = []

    for p in constructor.parameters:
        logger.info(f"Analyzing parameter: {p.name}")
        logger.info(f"  Type: {p.type}")

        if (
            hasattr(p.type, "type")
            and hasattr(p.type.type, "kind")
            and p.type.type.kind == "contract"
        ):
            res.append((p.name, "contract", p.name, [p.type.type.name]))

        elif hasattr(p.type, "name"):
            if p.type.name != "address":
                res.append((p.name, p.type.name, "YA_DO_NOT_KNOW", ["YA_DO_NOT_KNOW"]))
            else:
                res.append((p.name, p.type.name, [p.name], []))
        else:
            logger.warning(f"Unsupported parameter type: {p.name}")
            return None

    logger.info("Analyzing data flow inside constructor")

    for exps in constructor.expressions:
        if isinstance(exps, AssignmentOperation):
            exps_right = exps.expression_right
            exps_left = exps.expression_left

            if isinstance(exps_right, Identifier) and isinstance(exps_left, Identifier):
                for cst_param in res:
                    if (
                        isinstance(cst_param[2], list)
                        and exps_right.value.name in cst_param[2]
                    ):
                        cst_param[2].append(exps_left.value.name)

            elif isinstance(exps_right, TypeConversion) and isinstance(exps_left, Identifier):
                param_name, param_map_contract_name = extract_param_contract_map(exps_right)
                if param_name and param_map_contract_name:
                    for cst_param in res:
                        if (
                            isinstance(cst_param[2], list)
                            and param_name in cst_param[2]
                        ):
                            cst_param[3].append(param_map_contract_name)

        elif isinstance(exps, TypeConversion):
            param_name, param_map_contract_name = extract_param_contract_map(exps)
            if param_name and param_map_contract_name:
                for cst_param in res:
                    if (
                        isinstance(cst_param[2], list)
                        and param_name in cst_param[2]
                    ):
                        cst_param[3].append(param_map_contract_name)

    ret = []
    logger.info("=== Constructor parameter analysis result ===")

    for p_name, p_type, _, p_value in res:
        if p_type == "address" and len(p_value) == 0:
            p_value = ["YA_DO_NOT_KNOW"]

        p_value = list(set(p_value))
        assert len(p_value) == 1

        ret.append({
            "name": p_name,
            "type": p_type,
            "value": p_value[0]
        })

    logger.info("=== Constructor analysis finished ===")
    return ret


def extract_param_contract_map(exps: TypeConversion):
    inner_exp = exps.expression
    if (
        isinstance(inner_exp, Identifier)
        and isinstance(exps.type, UserDefinedType)
        and hasattr(exps.type, "type")
        and isinstance(exps.type.type, Contract)
    ):
        return inner_exp.value.name, exps.type.type.name

    return None, None
