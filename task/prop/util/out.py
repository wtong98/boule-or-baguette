"""Formatting proof for output"""
import json
import subprocess

from .data import *
from .proof import proof_has_failed

lean_repl_path = r'/home/grandpaa/workspace/imply/imply/task/prop/old/propositional_logic/random_gen/lean-repl'

# def get_thm_initial_state(num_vars: int, prop: Proposition) -> str:
#     tactic_state = StateTrackingCtx(num_vars, prop).get_cur_tactic_state()
#     return tactic_state


def format_example(n_atoms: int, prop: Proposition, proof: Proof) -> str:
    ctx = StateTrackingCtx(n_atoms, prop)
    traverse_proof(ctx, proof, 0)
    ret_text = "".join(ctx.result_buffer)

    if not proof_has_failed(proof):
        ret_text += "proof is complete\n"
    else:
        ret_text += "proof failed\n"

    return ret_text


class StateTrackingCtx:
    def __init__(self, num_vars: int, prop: Proposition, 
                 initial_indent:str = "  "):
        self.num_vars = num_vars
        self.prop = prop
        lean_text = to_lean_decl(num_vars, prop)
        self.states = [lean_text]
        self.cur_state_num = 0
        self.current_indent = initial_indent
        self.result_buffer : List[str] = []
        self.state_current_choices : Dict[int, int] = {}

    def push_state(self, state: str) -> None:
        """State should be a partial lean proof"""
        self.states.append(state)
        self.cur_state_num = len(self.states) - 1

    def get_cur_state_num(self) -> int:
        return self.cur_state_num

    def set_cur_state_num(self, num: int) -> None:
        self.cur_state_num = num

    def get_cur_state_text(self) -> str:
        return self.states[self.cur_state_num]

    def increment_indent(self) -> None:
        self.current_indent += "  "
    
    def decrement_indent(self) -> None:
        self.current_indent = self.current_indent[:-2]
    
    def get_cur_tactic_state(self) -> str:
        return get_lean_tactic_state(self.get_cur_state_text() + "\n" + self.current_indent + "sorry")


class ProofFailEscape(Exception): pass

def traverse_proof(ctx: StateTrackingCtx, proof: Proof, next_tactic_num: int):
    result = ctx.result_buffer
    def recurse(p: Proof):
        return traverse_proof(ctx, p, 0)
    def show_tactic(tactic: str, tactic_num: int) -> None:
        to_append_tactic_text = f"state_{ctx.get_cur_state_num()}_tactic_{tactic_num}:\n"
        assert to_append_tactic_text not in ctx.result_buffer
        result.append(to_append_tactic_text)
        result.append(tactic + "\n")
    def new_state_with_new_tactic(tactic: str) -> None:
        ctx.push_state(ctx.get_cur_state_text() + "\n" + ctx.current_indent + tactic)
    def repeat_previous_state() -> None:
        ctx.push_state(ctx.get_cur_state_text())
    def show_current_tactic_state() -> None:
        result.append(f"state_{ctx.get_cur_state_num()}:\n")
        result.append(ctx.get_cur_tactic_state() + "\n")
    
    def inversion_one_step(tactic: str, tactic_num: int) -> None:
        show_tactic(tactic, tactic_num)
        new_state_with_new_tactic(tactic)
        show_current_tactic_state()
    def one_step_with_increment(tactic: str, tactic_num: int) -> None:
        show_tactic(tactic, tactic_num)
        new_state_with_new_tactic(tactic)
        ctx.increment_indent()
        show_current_tactic_state()
    def decrement_and_repeat_previous_state() -> None:
        ctx.decrement_indent()
        repeat_previous_state()
        show_current_tactic_state()

    def process_choices(choices: List[Proof]) -> None:
        cur_state_num = ctx.get_cur_state_num()
        if cur_state_num not in ctx.state_current_choices:
            ctx.state_current_choices[cur_state_num] = 0

        cur_indent = ctx.current_indent
        for _, c in enumerate(choices):
            i = ctx.state_current_choices[cur_state_num]
            try:
                traverse_proof(ctx, c, i)
                break
            except ProofFailEscape:
                result.append(f"no solution, return to state {cur_state_num} [that leads to state {ctx.get_cur_state_num()}]\n")
                ctx.set_cur_state_num(cur_state_num)
                ctx.current_indent = cur_indent # reset indent
                show_current_tactic_state()
                ctx.state_current_choices[cur_state_num] += 1
                continue

    top_level_tactics = to_lean_tactic("", proof)
    if len(top_level_tactics) == 0:
        match proof:
            case NegAtomR(subproof):
                return recurse(subproof)
            case UpshiftR(subproof):
                return recurse(subproof)
            case PosTrueL(name, subproof):
                return recurse(subproof)
            case PosAtomL(name, subproof):
                return recurse(subproof)
            case DownshiftL(name, subproof):
                return recurse(subproof)
            case StableSeq(subproof):
                return recurse(subproof)
            case FocusR(subproof):
                return recurse(subproof)
            case FocusL(name, subproof):
                return recurse(subproof)
            case UpshiftL(name, subproof):
                return recurse(subproof)
            case DownshiftR(subproof):
                return recurse(subproof)
            case ProofFailed():
                raise ProofFailEscape()
            case FocusChoice(choices):
                process_choices(choices)
                return
            case OrR_choice(l, r):
                process_choices([l, r])
                return
            case NegAndL_choice(l, r):
                process_choices([l, r])
                return
            case _:
                raise ValueError(f"invalid proof: {proof}")

    if len(top_level_tactics) == 1:
        inversion_one_step(top_level_tactics[0], next_tactic_num)
        match proof:
            case ImpliesR((name, subproof)):
                recurse(subproof)
            case NegAndR(left, right):
                recurse(left)
                recurse(right)
            case NegTrueR():
                pass
            case FalseL(name):
                pass
            case OrR_left(subproof):
                recurse(subproof)
            case OrR_right(subproof):
                recurse(subproof)
            case PosAndR(left, right):
                recurse(left)
                recurse(right)
            case PosTrueR():
                pass
            case PosAtomR(name):
                pass
            case NegAndL_left(name, (left_name, left_proof)):
                recurse(left_proof)
            case NegAndL_right(name, (right_name, right_proof)):
                recurse(right_proof)
            case NegAtomL(name):
                pass
            case _:
                raise ValueError(f"invalid proof: {proof}")
    else:
        match proof:
            case OrL(name, (left_name, left_proof), (right_name, right_proof)):
                inversion_one_step(top_level_tactics[0], next_tactic_num)
                one_step_with_increment(top_level_tactics[1], 0)
                recurse(left_proof)
                decrement_and_repeat_previous_state()
                one_step_with_increment(top_level_tactics[2], 0)
                recurse(right_proof)
                decrement_and_repeat_previous_state()
            case PosAndL(name, (left_name, right_name, subproof)):
                inversion_one_step(top_level_tactics[0], next_tactic_num)
                inversion_one_step(top_level_tactics[1], 0)
                recurse(subproof)
            case ImpliesL(name, prop, left, (right_name, right_proof), additional_name):
                one_step_with_increment(top_level_tactics[0], next_tactic_num)
                recurse(left)
                decrement_and_repeat_previous_state()
                inversion_one_step(top_level_tactics[1], 0)
                recurse(right_proof)
            case _:
                raise ValueError(f"Invalid proof: {proof}")


def get_lean_tactic_state(lean_text: str) -> str:
    input_str = json.dumps({"cmd": lean_text})
    process = subprocess.Popen(["lake", "exec", "repl"], 
                               stdin=subprocess.PIPE, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True, 
                               cwd=lean_repl_path)
    try:
        stdout, _ = process.communicate(input_str)
        output_json = json.loads(stdout)

        if 'sorries' not in output_json or len(output_json["sorries"]) == 0:
            return "no goals"
        elif len(output_json["sorries"]) == 1:
            ret_text = output_json["sorries"][0]["goal"]
            if ret_text == "unknown goal":
                return "no goals"
            else:
                return ret_text
        else:
            raise ValueError(f"more than one goal: {output_json['sorries']}")

    finally:
        process.terminate()


def to_lean_decl(n_atoms : int, prop: Proposition) -> str:
    vars_decl = f"variable ({' '.join(f'p{i+1}' for i in range(n_atoms))} : Prop)\n" 
    theorem_text = f"example : {to_lean_theorem_text(prop)} := by\n"

    return vars_decl + theorem_text


def to_lean_theorem_text(p : Proposition):
        match p:
            case Atom(name):
                return name
            case PosAtom(name):
                return name
            case NegAtom(name):
                return name
            case And(left, right):
                return f"({to_lean_theorem_text(left)} ∧ {to_lean_theorem_text(right)})"
            case PosAnd(left, right):
                return f"({to_lean_theorem_text(left)} ∧ {to_lean_theorem_text(right)})"
            case NegAnd(left, right):
                return f"({to_lean_theorem_text(left)} ∧ {to_lean_theorem_text(right)})"
            case Or(left, right):
                return f"({to_lean_theorem_text(left)} ∨ {to_lean_theorem_text(right)})"
            case Implies(left, right):
                return f"({to_lean_theorem_text(left)} → {to_lean_theorem_text(right)})"
            case PTrue():
                return "True"
            case PFalse():
                return "False"
            case Downshift(operand):
                return to_lean_theorem_text(operand)
            case Upshift(operand):
                return to_lean_theorem_text(operand)
            case _:
                raise ValueError(f"Invalid proposition: {p}")


def to_lean_tactic(indent: str, proof: Proof) -> List[str]:
    match proof:
        case ImpliesR((name, subproof)):
            return [f"{indent}intro {name}"]
        case NegAndR(left, right):
            return [f"{indent}apply And.intro"]
        case NegTrueR():
            return [f"{indent}apply True.intro"]
        case NegAtomR(subproof):
            return []
        case UpshiftR(subproof):
            return []
        case OrL(name, (left_name, left_proof), (right_name, right_proof)):
            return [f"{indent}cases {name}",
                   f"{indent}case inl {left_name} =>",
                     f"{indent}case inr {right_name} =>"]
        case FalseL(name):
            return [f"{indent}apply False.elim {name}"]
        case PosAndL(name, (left_name, right_name, subproof)):
            return [f"{indent}let {left_name} := {name}.left",
                   f"{indent}let {right_name} := {name}.right"]
        case PosTrueL(name, subproof):
            return []
        case PosAtomL(name, subproof):
            return []
        case DownshiftL(name, subproof):
            return []
        case StableSeq(subproof):
            return []
        case FocusR(subproof):
            return []
        case FocusL(name, subproof):
            return []
        case OrR_left(subproof):
            return [f"{indent}apply Or.inl"]
        case OrR_right(subproof):
            return [f"{indent}apply Or.inr"]
        case PosAndR(left, right):
            return [f"{indent}apply And.intro"]
        case PosTrueR():
            return [f"{indent}apply True.intro"]
        case PosAtomR(name):
            return [f"{indent}exact {name}"]
        case DownshiftR(subproof):
            return []
        case ImpliesL(name, prop, left, (right_name, right_proof), additional_name):
            return [f"{indent}have {additional_name} : {to_lean_theorem_text(prop)} := by",
                   f"{indent}let {right_name} := {name} {additional_name}"]
        case NegAndL_left(name, (left_name, left_proof)):
            return [f"{indent}let {left_name} := {name}.left"]
        case NegAndL_right(name, (right_name, right_proof)):
            return [f"{indent}let {right_name} := {name}.right"]
        case NegAtomL(name):
            return [f"{indent}exact {name}"]
        case UpshiftL(name, subproof):
            return []
        case ProofFailed():
            return []
        case FocusChoice(_):
            return []
        case NegAndL_choice(_, _):
            return []
        case OrR_choice(_, _):
            return []
        case _:
            raise ValueError(f"Invalid proof: {proof}")