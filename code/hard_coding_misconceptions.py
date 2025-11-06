import re
import json
import math
import inspect
from typing import Any, Iterable, Tuple, List, Optional, Dict
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
def _normalize(s: str) -> str:
    """Normalize minus and remove surrounding whitespace."""
    if s is None:
        return ""
    s = s.strip()
    # Replace unicode minus and common dashes with '-'
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    return s


def _find_matching_paren(s: str, open_idx: int) -> Optional[int]:
    """Find the matching ')' for the '(' at open_idx, respecting nesting."""
    assert s[open_idx] == "("
    depth = 0
    for i in range(open_idx, len(s)):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    return None

def _split_top_level_terms(expr: str) -> List[str]:
    """
    Split by top-level '+' and '-' only (outside parentheses), and keep
    the sign with the following term. '*' and '/' never split.
    NEW: If the '+' or '-' is immediately preceded by '*' or '/',
         treat it as a unary sign and DO NOT split there.
    """
    expr = _normalize(re.sub(r"\s+", "", expr))
    if expr == "":
        return []

    terms: List[str] = []
    depth = 0
    cur = ""
    for i, ch in enumerate(expr):
        if ch in "([{":
            depth += 1
            cur += ch
        elif ch in ")]}":
            depth -= 1
            cur += ch
        elif depth == 0 and ch in "+-":
            prev = expr[i-1] if i > 0 else ""
            if prev in "*/":
                # unary +/− after * or / -> keep with current factor
                cur += ch
            else:
                if cur:
                    terms.append(cur)
                    cur = ch  # start new term with its sign
                else:
                    # expression starts with +/−
                    cur = ch
        else:
            cur += ch

    if cur:
        terms.append(cur)

    return terms

def _extract_lhs_multiplier(lhs: str) -> Tuple[Optional[str], str]:
    """
    If LHS begins with something like 'k( ... )' or '+( ... )' or '-( ... )'
    possibly with a '*', extract the multiplier and return (multiplier, core_lhs_without_that_prefix).
    - '3(2x+1)' => ('3', '(2x+1)')
    - '2*(x+4)+5' => ('2', '(x+4)+5')
    - '-(x+2)-3' => ('-1', '(x+2)-3')
    Otherwise => (None, lhs)
    """
    s = _normalize(re.sub(r"\s+", "", lhs))
    if not s:
        return None, lhs

    # Optional sign at the very start
    sign = ""
    if s[0] in "+-":
        sign = s[0]
        s_wo_sign = s[1:]
    else:
        s_wo_sign = s

    # Look for first '(' at top-level (after optional factor and optional '*')
    # We only allow characters before '(' that don't include '+' or '-'.
    # Accept digits, letters, '.', '*'
    # Pattern:  ^([A-Za-z0-9.*]*)\((...)
    if not s_wo_sign or s_wo_sign[0] != "(":
        # We might have factor then '(':
        # Find the first '(' that is not after any '+' or '-' in the prefix
        paren_idx = s_wo_sign.find("(")
        if paren_idx == -1:
            return None, lhs  # no parentheses
        # If there's a '+' or '-' before '(', it's not a pure leading multiplier
        if any(op in s_wo_sign[:paren_idx] for op in "+-"):
            return None, lhs
        # Leading "factor" (may include '*')
        factor = s_wo_sign[:paren_idx]
        # If there's a factor with '*' or alphanum (not empty), OK
        if not factor:
            # e.g., starts with '(', no explicit multiplier; treat as None
            return None, lhs
        # OK: treat as multiplier; drop trailing '*' if present
        if factor.endswith("*"):
            factor = factor[:-1]
        # If factor is just sign replacement '+' or '-', turn into +1/-1 below (handled next)
    else:
        # It starts with '(' immediately after optional sign; treat sign as multiplier ±1
        factor = ""

    # Match the parentheses span
    full = s if sign == "" else sign + s_wo_sign
    # Locate the first '(' in full after optional sign/factor we already reasoned about
    first_paren = full.find("(") if sign == "" else sign.__len__() + s_wo_sign.find("(")
    close_paren = _find_matching_paren(full, first_paren)
    if close_paren is None:
        return None, lhs  # unbalanced; fallback

    # Build multiplier text
    if factor == "" and sign in {"+", "-"}:
        multiplier = "1" if sign == "+" else "-1"
    else:
        base = (("-" if sign == "-" else "") + factor) if factor else (sign if sign else "")
        multiplier = base if base not in {"+", ""} else "1"
        # Normalize "+2" -> "2"
        if multiplier.startswith("+"):
            multiplier = multiplier[1:]

    core_lhs = full[first_paren: ]  # '( ... )...' (we keep the rest; later terms splitter will handle any + or - outside)

    # Rebuild a pretty 'lhs' string returning same formatting as input for downstream splitting
    return multiplier, core_lhs

def _as_list(x: Any) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except Exception:
                return []
        return [] if s == "" or s.lower() == "nan" else [s]
    return [] if pd.isna(x) else [str(x)]

def _to_set(xs: Iterable[str]) -> set:
    return {t.strip() for t in xs if isinstance(t, str) and t.strip() != ""}

def _jsonify_list_cols_if_needed(df: pd.DataFrame, step_col: str, want_json: bool, used_json_kw: bool) -> pd.DataFrame:
    """If annotate_dataframe lacked json_lists support, JSON-encode list-valued columns now."""
    if not want_json or used_json_kw:
        return df
    for suffix in ("lhs_terms", "rhs_terms", "lhs_paren_terms"):
        col = f"{step_col}__{suffix}"
        if col in df.columns and df[col].map(lambda v: isinstance(v, list)).any():
            df[col] = df[col].apply(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
    return df

@dataclass
class ParsedEquation:
    lhs_terms: List[str]
    rhs_terms: List[str]
    lhs_multiplier: Optional[str]  # e.g., '3', '-1', None
    lhs_paren_terms: Optional[List[str]]  # terms INSIDE the first bracket if multiplier extracted; else None


# ---------- New: equation row filtering ----------

def annotate_dataframe(df: pd.DataFrame, step_col: str) -> pd.DataFrame:
    """
    For each row, parse df[step_col] and add:
      - f"{step_col}__lhs_terms" (list[str])
      - f"{step_col}__rhs_terms" (list[str])
      - f"{step_col}__lhs_multiplier" (str or None)
      - f"{step_col}__lhs_paren_terms" (list[str] or None)
    """
    def _safe_parse(x: Any) -> Dict[str, Any]:
        parsed = parse_equation(str(x) if pd.notna(x) else "")
        return {
            f"{step_col}__lhs_terms": parsed.lhs_terms,
            f"{step_col}__rhs_terms": parsed.rhs_terms,
            f"{step_col}__lhs_multiplier": parsed.lhs_multiplier,
            f"{step_col}__lhs_paren_terms": parsed.lhs_paren_terms,
        }

    parsed_df = df[step_col].apply(_safe_parse).apply(pd.Series)
    # If you need to persist to CSV later, you might want JSON strings instead of raw lists:
    # for c in parsed_df.columns:
    #     if parsed_df[c].map(lambda v: isinstance(v, list)).any():
    #         parsed_df[c] = parsed_df[c].map(lambda v: json.dumps(v, ensure_ascii=False))
    return pd.concat([df, parsed_df], axis=1)



def _canon_term(t: Any) -> str:
    """
    Canonicalize a term for set comparisons:
      - strip spaces, normalize unicode minus to '-'
      - lowercase 'X' -> 'x'
      - ensure an explicit leading sign; if missing, prepend '+'
    Examples: '3' -> '+3', '2x' -> '+2x', '+2x' -> '+2x', '-x' -> '-x'
    """
    s = "" if t is None else str(t)
    s = _normalize(re.sub(r"\s+", "", s))
    s = s.replace("X", "x")
    if not s:
        return s
    if s[0] not in "+-":
        s = "+" + s
    return s

def _canon_term(t: str) -> str:
    """Canonicalize a token for movement comparison:
       - strip whitespace
       - normalize unicode minus
       - drop a single leading '+'
       - lower-case 'X' -> 'x'
    """
    if t is None:
        return ""
    s = _normalize(str(t))  # trims, normalizes minus
    if not s:
        return ""
    if s.startswith("+"):
        s = s[1:]
    return s.replace("X", "x")

def _to_canon_set(xs: Iterable[str]) -> set:
    """Set of canonicalized, non-empty terms."""
    out = set()
    for t in _as_list(xs):
        if isinstance(t, str):
            tt = _canon_term(t)
            if tt:
                out.add(tt)
        elif t is not None and not (isinstance(t, float) and math.isnan(t)):
            out.add(_canon_term(str(t)))
    return out

def parse_equation(equation: str) -> ParsedEquation:
    """
    Decompose an equation like '2x + 3 = 5x + 7' or '3(2x+1) = 7' per the rules.
    """
    eq = _normalize(equation or "")
    parts = eq.split("=")
    lhs = parts[0] if parts else ""
    rhs = parts[1] if len(parts) > 1 else ""

    # Try multiplier extraction on LHS
    mult, lhs_after = _extract_lhs_multiplier(lhs)
    lhs_after_stripped = _normalize(lhs_after)

    # If we have something like '(ax+b)+c', we want bracket terms + outside +/-
    lhs_terms: List[str] = []
    lhs_paren_terms: Optional[List[str]] = None
    if lhs_after_stripped.startswith("(") and mult is not None:
        # isolate the first (...) block
        close_idx = _find_matching_paren(lhs_after_stripped, 0)
        inside = lhs_after_stripped[1:close_idx] if close_idx is not None else lhs_after_stripped[1:]
        rest = lhs_after_stripped[close_idx+1:] if close_idx is not None else ""
        lhs_paren_terms = _split_top_level_terms(inside)
        lhs_terms = list(lhs_paren_terms)  # copy
        # Add any trailing + / - terms after the closing parenthesis
        if rest:
            lhs_terms.extend(_split_top_level_terms(rest))
    else:
        lhs_terms = _split_top_level_terms(lhs)
        mult = None  # only store multiplier when we actually had bracketed product at the left

    rhs_terms = _split_top_level_terms(rhs)

    return ParsedEquation(lhs_terms=lhs_terms, rhs_terms=rhs_terms,
                          lhs_multiplier=mult, lhs_paren_terms=lhs_paren_terms)


# -----------

def _call_annotate(df: pd.DataFrame, step_col: str, save_lists_as_json: bool) -> tuple[pd.DataFrame, bool]:
    """
    Call annotate_dataframe with or without json_lists depending on its signature.
    If annotate_dataframe is not available, but parse_equation is, a small fallback is used.
    """
    if 'annotate_dataframe' in globals():
        if 'json_lists' in inspect.signature(annotate_dataframe).parameters:
            return annotate_dataframe(df, step_col, json_lists=save_lists_as_json), True
        else:
            return annotate_dataframe(df, step_col), False
    elif 'parse_equation' in globals():
        # Fallback: create the columns directly via parse_equation
        def _safe_parse(x: Any):
            p = parse_equation("" if pd.isna(x) else str(x))
            return {
                f"{step_col}__lhs_terms": p.lhs_terms,
                f"{step_col}__rhs_terms": p.rhs_terms,
                f"{step_col}__lhs_multiplier": p.lhs_multiplier,
                f"{step_col}__lhs_paren_terms": p.lhs_paren_terms,
            }
        df = pd.concat([df, df[step_col].apply(_safe_parse).apply(pd.Series)], axis=1)
        return df, False
    else:
        raise RuntimeError("Neither annotate_dataframe nor parse_equation found in scope.")


_ALLOWED_CHARS_RE = re.compile(r'^[0-9xX+\-*/().=\s]+$')

def _exactly_one_equals(s: str) -> bool:
    return s.count("=") == 1

def _equation_has_only_allowed_chars(s: str) -> bool:
    return bool(_ALLOWED_CHARS_RE.fullmatch(s))


def filter_valid_equations(
    df: pd.DataFrame,
    col_from: str = "step_from",
    col_to: str   = "step_to",
    report: bool = True,
    implied_col: str = "implied",
    output_dir: str = "analysis_reports"
) -> pd.DataFrame:
    """
    Drop rows where either equation side:
      - doesn't contain exactly one '=', OR
      - contains illegal characters.
    Also:
      - strips "solve:" prefix
      - saves dropped rows to a report file
      - counts when dropped rows are same as prev/next step (or same after removing x=)
        but only if implied != 1.0
    """
    Path(output_dir).mkdir(exist_ok=True)
    invalid_path = Path(output_dir) / "invalid_equations_report.txt"

    def _clean_prefix(s: str) -> str:
        s = str(s)
        if s.lower().startswith("solve:"):
            return s.split("solve:", 1)[1].strip()
        return s.strip()

    def _is_valid(eq: Any) -> bool:
        if pd.isna(eq):
            return False
        s = _clean_prefix(str(eq))
        return _exactly_one_equals(s) and _equation_has_only_allowed_chars(s)

    # Apply prefix cleanup before validation
    df[col_from] = df[col_from].astype(str).apply(_clean_prefix)
    df[col_to]   = df[col_to].astype(str).apply(_clean_prefix)

    mask_from = df[col_from].map(_is_valid)
    mask_to   = df[col_to].map(_is_valid)
    mask = mask_from & mask_to
    dropped_mask = ~mask

    dropped_rows = df.loc[dropped_mask].copy()
    dropped = dropped_rows.shape[0]
    kept = int(mask.sum())

    if report:
        print(f"[filter_valid_equations] Dropped {dropped} rows; kept {kept}.")

    # --- Detect same or x= same steps ---
    if dropped > 0:
        # Only rows with implied != 1.0
        if implied_col in df.columns:
            not_implied_mask = df[implied_col].astype(str).str.strip() != "1.0"
            dropped_rows = dropped_rows.loc[not_implied_mask]

        def _strip_xeq(s: str) -> str:
            s = str(s)
            return s[2:].strip() if s.startswith("x=") else s

        dropped_rows["same_step"] = (
            (dropped_rows[col_from] == dropped_rows[col_to]) |
            (_strip_xeq(dropped_rows[col_from]) == _strip_xeq(dropped_rows[col_to]))
        )

        same_count = int(dropped_rows["same_step"].sum())

        # Save detailed report
        with open(invalid_path, "w", encoding="utf-8") as f:
            f.write("=== INVALID EQUATIONS REPORT ===\n")
            f.write(f"Total invalid rows: {dropped}\n")
            f.write(f"Of which same/‘x=’same steps (non-implied): {same_count}\n\n")
            f.write(dropped_rows[[col_from, col_to, "same_step"]].to_string(index=False))
            f.write("\n")

        if report:
            print(f"Saved invalid equation details → {invalid_path}")

    return df.loc[mask].reset_index(drop=True)

# ---------- Existing helpers (tweaked _sum_numeric_terms) ----------

_NUMERIC_RE = re.compile(r'^[+-]?\d+(?:\.\d+)?$')

def _as_list(x: Any) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s)
                return v if isinstance(v, list) else [s]
            except Exception:
                return []
        return [] if s == "" or s.lower() == "nan" else [s]
    return [] if (x is None or (isinstance(x, float) and math.isnan(x))) else [str(x)]

# --- put this with your helpers (right above _to_set) ---
def _normalize_term(t: str) -> str:
    """Canonicalize tokens for set-comparisons:
       - strip whitespace
       - drop a single leading '+' (so '+3' == '3', '+x' == 'x')
    """
    if not isinstance(t, str):
        t = str(t)
    t = t.strip()
    if t.startswith("+"):
        t = t[1:]
    return t

# --- REPLACE your existing _to_set used by add_all_todos_columns with this ---
def _to_set(xs: Iterable[str]) -> set:
    return {_normalize_term(t) for t in xs
            if isinstance(t, str) and _normalize_term(t) != ""}
def _has_x(term: str) -> bool:
    return isinstance(term, str) and ('x' in term.lower())

def _is_numeric_term(term: str) -> bool:
    return isinstance(term, str) and bool(_NUMERIC_RE.fullmatch(term.strip()))

def _format_signed_number(val: float) -> str:
    if val is None or math.isnan(val):
        return ""
    if abs(val - int(val)) < 1e-12:
        n = int(val)
        return f"{n:+d}"
    return f"{val:+g}"

def _is_signed_zero_token(t: Any) -> bool:
    """True iff token is exactly '+0' or '-0' after whitespace/minus normalization."""
    s = _normalize(str(t))
    return s in {"+0", "-0"}

def _sum_numeric_terms(terms: Iterable[str]) -> Tuple[str, float]:
    """
    Sum numeric tokens. Treat bare 'x'/' +x'/'-x' as ±1 (per your request),
    but ignore coefficiented x like '2x', '-0.5x', '3*x', etc.
    """
    total = 0.0
    for t in terms:
        ts = str(t).strip()
        # Handle bare x as ±1
        if ts.lower() in {"x", "+x"}:
            total += 1.0
            continue
        if ts.lower() == "-x":
            total -= 1.0
            continue
        # Ignore any other x-containing token
        if _has_x(ts):
            continue
        # Pure numeric
        if _is_numeric_term(ts):
            total += float(ts)
    return _format_signed_number(total), total


def _norm_flat(s: str) -> str:
    """Normalize for string comparisons: strip spaces, normalize minus/dashes."""
    return _normalize(re.sub(r"\s+", "", s or ""))

def _lhs_raw(eq: str) -> str:
    """Return raw LHS (text before '=') or '' if malformed/absent."""
    if not isinstance(eq, str):
        return ""
    parts = eq.split("=")
    return parts[0] if parts else ""


def _lhs_raw(step_expr: str) -> str:
    """Return the left-hand side of a full equation string, after normalizing and stripping any leading 'solve:'."""
    s = _normalize(str(step_expr or ""))
    s = re.sub(r'^\s*solve\s*:\s*', '', s, flags=re.IGNORECASE)
    return s.split("=", 1)[0] if "=" in s else s

def _norm_flat(s: str) -> str:
    """Tight normalization for equality checks (remove spaces, normalize minus, keep case-lowered x)."""
    if s is None:
        return ""
    s = _normalize(re.sub(r"\s+", "", str(s)))
    return s.replace("X", "x")
def _multiplier_drop_flag(step_from_expr: str, step_to_expr: str) -> tuple[bool, str]:
    """
    Detect 'multiplier drop' on the LHS using full equations:
      If step_to LHS equals step_from LHS with the leading multiplier-and-star before the first
      parenthesis removed (keeping the parentheses), OR equals the same with that first pair
      of parentheses dropped, flag True.

    Returns: (flag, variant) where variant in {"ignore_prefix","drop_parens",""}.
    """
    lhs_from = _lhs_raw(step_from_expr)
    lhs_to   = _lhs_raw(step_to_expr)
    if not lhs_from or not lhs_to:
        return False, ""

    mult, core = _extract_lhs_multiplier(lhs_from)  # core starts at '(' if found; else (None, lhs_from)
    a, b = _extract_lhs_multiplier("(17x-6)")
    assert a == None or "" or 1, f"Unexpected parse in _multiplier_drop_flag test: {a}"
    a, b = _extract_lhs_multiplier("4(17x-6)")
    assert a == "4", f"Unexpected parse in _multiplier_drop_flag test: {a}"
    if mult == "1" or None or mult == "":
        return False, ""  # no leading multiplier pattern like k*( ... ) or ±( ... )

    core = _normalize(core)
    if not core or core[0] != "(":
        return False, ""

    close_idx = _find_matching_paren(core, 0)
    if close_idx is None:
        return False, ""  # unbalanced

    inside = core[1:close_idx]
    rest   = core[close_idx+1:]  # trailing +... after the first ')', if any

    # Variant A: just ignore the multiplier prefix (keep outer parens)
    cand_keep_parens = core
    # Variant B: also drop the outer parens
    cand_drop_parens = inside + rest

    lhs_to_norm = _norm_flat(lhs_to)
    if lhs_to_norm == _norm_flat(cand_keep_parens):
        return True, "ignore_prefix"
    if lhs_to_norm == _norm_flat(cand_drop_parens):
        return True, "drop_parens"

    return False, ""

def _canon_abs_token(t: Any) -> str:
    """Canonicalize and then drop a single leading '-' so comparisons ignore sign."""
    s = _canon_term(t)
    return s[1:] if s.startswith("-") else s
# ---------- Movement + sums + combined-unlike (unchanged semantics) ----------

def add_all_todos_columns(df: pd.DataFrame,
                          col_from: str = "step_from",
                          col_to: str   = "step_to",
                          save_lists_as_json: bool = False,
                          drop_bad_rows: bool = False,
                          report_drops: bool = True) -> pd.DataFrame:
    """
    Full pipeline:
      - (optional) filter out malformed equations
      - add term lists via annotate/parse
      - movement flags
      - numeric sums with bare x→±1
      - combined-unlike detection
    """
    if drop_bad_rows:
        df = filter_valid_equations(df, col_from=col_from, col_to=col_to, report=report_drops)

    # annotate (same adapter you already had)
    df, used_json_kw_from = _call_annotate(df, col_from, save_lists_as_json)
    df, used_json_kw_to   = _call_annotate(df, col_to,   save_lists_as_json)
    df = _jsonify_list_cols_if_needed(df, col_from, save_lists_as_json, used_json_kw_from)
    df = _jsonify_list_cols_if_needed(df, col_to,   save_lists_as_json, used_json_kw_to)

    FROM_LHS = f"{col_from}__lhs_terms"
    FROM_RHS = f"{col_from}__rhs_terms"
    TO_LHS   = f"{col_to}__lhs_terms"
    TO_RHS   = f"{col_to}__rhs_terms"

    moved_lr_list, moved_rl_list = [], []
    moved_lr_flag, moved_rl_flag = [], []
    dup_lhs_flags, dup_rhs_flags = [], []
    dup_lhs_terms, dup_rhs_terms = [], []
    lhs_from_sums_fmt, rhs_from_sums_fmt = [], []
    lhs_from_sums_val, rhs_from_sums_val = [], []
    combined_unlike_lhs_to_flags, combined_unlike_rhs_to_flags = [], []
    bracket_expansion_flags = []
    multiplier_drop_flags =[]
    multiplier_drop_flags = []
    multiplier_drop_variants = []
    multiplier_to_addition_flags = []
    for _, row in df.iterrows():
        lhs_from = _as_list(row.get(FROM_LHS, []))
        rhs_from = _as_list(row.get(FROM_RHS, []))
        lhs_to   = _as_list(row.get(TO_LHS,   []))
        rhs_to   = _as_list(row.get(TO_RHS,   []))


                # Canonicalized token sets so '+33' ≡ '33' and '+2x' ≡ '2x'
        s_lhs_from = _to_canon_set(lhs_from)
        s_rhs_from = _to_canon_set(rhs_from)
        s_lhs_to   = _to_canon_set(lhs_to)
        s_rhs_to   = _to_canon_set(rhs_to)
        trivial_x_to_x_lhs = (s_lhs_from == {"x"} and s_rhs_to == {"x"}) 
        trivial_x_to_x_rhs = (s_rhs_from == {"x"} and s_lhs_to == {"x"})
        raw_l2r = sorted([t for t in s_lhs_from if (t in s_rhs_to) and (t not in s_rhs_from)])
        raw_r2l = sorted([t for t in s_rhs_from if (t in s_lhs_to) and (t not in s_lhs_from)])

        # --- DUPES: same token also appears on the same side in step_to ---
        dupes_on_lhs_to = sorted([t for t in raw_l2r if t in s_lhs_to])
        dupes_on_rhs_to = sorted([t for t in raw_r2l if t in s_rhs_to])

        # store notices if you like
        dup_lhs_flags.append(bool(dupes_on_lhs_to))
        dup_rhs_flags.append(bool(dupes_on_rhs_to))
        dup_lhs_terms.append(dupes_on_lhs_to)
        dup_rhs_terms.append(dupes_on_rhs_to)

        # --- FINAL moves = RAW minus DUPES (do NOT count moves when dupes exist) ---
        moved_lhs_to_rhs = [t for t in raw_l2r if t not in dupes_on_lhs_to]
        moved_rhs_to_lhs = [t for t in raw_r2l if t not in dupes_on_rhs_to]

        dup_lhs_flags.append(bool(dupes_on_lhs_to))
        dup_rhs_flags.append(bool(dupes_on_rhs_to))
        dup_lhs_terms.append(dupes_on_lhs_to)
        dup_rhs_terms.append(dupes_on_rhs_to)

        if dupes_on_lhs_to:
            print(f"\033[91m➜ DUPLICATE on LHS_to: {dupes_on_lhs_to} | "
                f"from: {row.get(col_from, '')} | to: {row.get(col_to, '')}\033[0m")
        if dupes_on_rhs_to:
            print(f"\033[91m➜ DUPLICATE on RHS_to: {dupes_on_rhs_to} | "
                f"from: {row.get(col_from, '')} | to: {row.get(col_to, '')}\033[0m")
        # (A) Exact movement flags on canonical tokens

        # Rule (1): If step_from LHS is exactly {'+x'} and step_to RHS is exactly {'+x'},
        #           do NOT count a LHS→RHS move.
        lhs_from_only_x = (s_lhs_from == {"+x"})
        rhs_to_only_x   = (s_rhs_to   == {"+x"})
        if lhs_from_only_x and rhs_to_only_x:
            moved_lhs_to_rhs = []
        moved_lr_list.append(moved_lhs_to_rhs)
        moved_rl_list.append(moved_rhs_to_lhs)
        moved_lr_flag.append(len(moved_lhs_to_rhs) > 0)
        moved_rl_flag.append(len(moved_rhs_to_lhs) > 0)

        # Numeric sums on step_from (now counts bare x as ±1)
        lhs_sum_fmt, lhs_sum_val = _sum_numeric_terms(lhs_from)
        rhs_sum_fmt, rhs_sum_val = _sum_numeric_terms(rhs_from)
        lhs_from_sums_fmt.append(lhs_sum_fmt)
        rhs_from_sums_fmt.append(rhs_sum_fmt)
        lhs_from_sums_val.append(lhs_sum_val)
        rhs_from_sums_val.append(rhs_sum_val)

        # Combined-unlike detection
        lhs_from_has_x   = any(_has_x(t) for t in s_lhs_from)
        lhs_from_has_num = any(_is_numeric_term(t) or (str(t).strip().lower() in {"x","+x","-x"})
                               for t in s_lhs_from)
        rhs_from_has_x   = any(_has_x(t) for t in s_rhs_from)
        rhs_from_has_num = any(_is_numeric_term(t) or (str(t).strip().lower() in {"x","+x","-x"})
                               for t in s_rhs_from)

        # just before the checks:
        lhs_sum_token   = _canon_term(lhs_sum_fmt) if lhs_sum_fmt else ""
        lhs_sum_x_token = _canon_term(lhs_sum_fmt + "x") if lhs_sum_fmt else ""
        rhs_sum_token   = _canon_term(rhs_sum_fmt) if rhs_sum_fmt else ""
        rhs_sum_x_token = _canon_term(rhs_sum_fmt + "x") if rhs_sum_fmt else ""

        # and then:
        ((lhs_sum_token and lhs_sum_token in s_lhs_to) or
        (lhs_sum_x_token and lhs_sum_x_token in s_lhs_to))
        # same for RHS

        combined_unlike_lhs_to = (
            lhs_from_has_x and lhs_from_has_num and
            ((lhs_sum_token and lhs_sum_token in s_lhs_to) or
             (lhs_sum_x_token and lhs_sum_x_token in s_lhs_to))
        )
        combined_unlike_rhs_to = (
            rhs_from_has_x and rhs_from_has_num and
            ((rhs_sum_token and rhs_sum_token in s_rhs_to) or
             (rhs_sum_x_token and rhs_sum_x_token in s_rhs_to))
        )

        combined_unlike_lhs_to_flags.append(bool(combined_unlike_lhs_to))
        combined_unlike_rhs_to_flags.append(bool(combined_unlike_rhs_to))

                # --- NEW: bracket-expansion mistake detection ---
        # Goal: True iff
        #  (a) step_from has brackets (we have lhs_paren_terms), AND
        #  (b) step_to has NO brackets, AND
        #  (c) implied != 1.0, AND
        #  (d) (multiplier * first_term_in_bracket) is found on LHS(step_to), AND
        #  (e) the second term from inside the bracket still appears on LHS(step_to) unchanged
        lhs_mult = str(row.get(f"{col_from}__lhs_multiplier", "") or "").strip()
        lhs_paren_terms_list = _as_list(row.get(f"{col_from}__lhs_paren_terms", []))

        mult_is_one = False
        if lhs_mult is not None and lhs_mult != "":
            try:
                mult_is_one = (abs(float(lhs_mult) - 1.0) < 1e-12)
            except Exception:
                mult_is_one = False

        second_is_signed_zero = False
        if lhs_paren_terms_list and len(lhs_paren_terms_list) > 1:
            second_is_signed_zero = _is_signed_zero_token(lhs_paren_terms_list[-1])

        has_brackets_from = bool(lhs_paren_terms_list)
        step_to_str = str(row.get(col_to, "") or "")
        has_brackets_to = ("(" in step_to_str) or (")" in step_to_str)

        implied_is_one = str(row.get("implied", "")).strip() == "1.0"

        # Default
        bracket_expansion_flag = False

        if has_brackets_from and (not has_brackets_to) and (not implied_is_one) and lhs_mult:
            # Need at least two terms inside the bracket to check "second term unchanged"
            if len(lhs_paren_terms_list) >= 2:
                first_term  = lhs_paren_terms_list[0]
                second_term = lhs_paren_terms_list[1]

                # Build candidate products for (multiplier * first_term)
                # We check several canonicalized forms: simplified when possible, plus unsimplified fallbacks.
                cands = set()
                ft = _canon_term(first_term)
                m  = _canon_term(lhs_mult)

                # Always try unsimplified spellings the student might literally write
                cands.add(_canon_term(f"{m}*{ft}"))
                cands.add(_canon_term(f"{m}{ft}"))

                # Try simplified numeric*monomial or numeric*number
                try:
                    m_val = float(m)
                except Exception:
                    m_val = None

                import re as _re_local

                # Case: first term is like ([+/-]?)(coeff?)x
                m1 = _re_local.fullmatch(r'([+-])?(\d+(?:\.\d+)?)?x', ft)
                if (m_val is not None) and m1:
                    sign  = -1.0 if (m1.group(1) == '-') else 1.0
                    coeff = float(m1.group(2)) if m1.group(2) else 1.0
                    prod  = m_val * sign * coeff
                    if abs(prod - 1.0) < 1e-12:
                        simplified = "x"
                    elif abs(prod + 1.0) < 1e-12:
                        simplified = "-x"
                    else:
                        simplified = f"{_format_signed_number(prod).lstrip('+')}x"
                    cands.add(_canon_term(simplified))

                # Case: first term is purely numeric
                m2 = _re_local.fullmatch(r'([+-])?(\d+(?:\.\d+)?)$', ft)
                if (m_val is not None) and m2:
                    sign  = -1.0 if (m2.group(1) == '-') else 1.0
                    coeff = float(m2.group(2))
                    prod  = m_val * sign * coeff
                    simplified = _format_signed_number(prod).lstrip('+')
                    cands.add(_canon_term(simplified))

                s_lhs_to_abs = {_canon_abs_token(t) for t in s_lhs_to}

                # (d) product candidate present on LHS(step_to)?
                product_ok = any(_canon_abs_token(c) in s_lhs_to_abs for c in cands)

                # (e) second term from bracket appears on LHS(step_to), ignoring sign?
                second_ok = (_canon_abs_token(second_term) in s_lhs_to_abs)

                bracket_expansion_flag = bool(
                    product_ok and second_ok and not second_is_signed_zero and not mult_is_one
                )

        bracket_expansion_flags.append(bracket_expansion_flag)

        step_from_expr = str(row.get(col_from, "") or "")
        step_to_expr   = str(row.get(col_to,   "") or "")

        # --- Multiplier drop detection (LHS) ---
        mflag, mvar = _multiplier_drop_flag(step_from_expr, step_to_expr)
        multiplier_drop_flags.append(bool(mflag))
        multiplier_drop_variants.append(mvar)
    # Output columns
    df["which_moved_lhs_to_rhs_exact"] = (
        [json.dumps(x, ensure_ascii=False) for x in moved_lr_list] if save_lists_as_json else moved_lr_list
    )
    df["which_moved_rhs_to_lhs_exact"] = (
        [json.dumps(x, ensure_ascii=False) for x in moved_rl_list] if save_lists_as_json else moved_rl_list
    )
    df["moved_lhs_to_rhs_exact"] = moved_lr_flag
    df["moved_rhs_to_lhs_exact"] = moved_rl_flag
    df["at_least_one_moved_over"] = [a or b for a, b in zip(moved_lr_flag, moved_rl_flag)]
    df["bracket_expansion_flags"] = bracket_expansion_flags 
    # or multiplier_drop_flags
    df["multiplier_drop_flag"] = multiplier_drop_flags
    df["distributive_property_mistake"] = (
            df["bracket_expansion_flags"] | df["multiplier_drop_flag"]
        )
    # df["multiplier_drop_variant"] = multiplier_drop_variants
    df[f"{col_from}__lhs_numeric_sum"] = lhs_from_sums_fmt
    df[f"{col_from}__rhs_numeric_sum"] = rhs_from_sums_fmt
    df[f"{col_from}__lhs_numeric_sum_val"] = lhs_from_sums_val
    df[f"{col_from}__rhs_numeric_sum_val"] = rhs_from_sums_val

    df["combined_unlike_on_lhs_to"] = combined_unlike_lhs_to_flags
    df["combined_unlike_on_rhs_to"] = combined_unlike_rhs_to_flags
    df["At_least_one_combined_unlike"] = [
        a or b for a, b in zip(combined_unlike_lhs_to_flags, combined_unlike_rhs_to_flags)
    ]

    return df


def run_todos_tests(verbosity: int = 2) -> None:
    """
    Run a compact unittest suite for the TODOS pipeline and print a summary
    with pass proportions.

    Preconditions: the following callables must be available in scope:
        - add_all_todos_columns(df, col_from="step_from", col_to="step_to",
                                save_lists_as_json=False, drop_bad_rows=False, report_drops=True)
        - filter_valid_equations(df, col_from="step_from", col_to="step_to", report=True)
    """
    import unittest
    import pandas as pd

    # ---- Sanity guardrails ---------------------------------------------------
    missing = []
    if "add_all_todos_columns" not in globals():
        missing.append("add_all_todos_columns")
    if "filter_valid_equations" not in globals():
        missing.append("filter_valid_equations")
    if missing:
        raise RuntimeError(
            f"Missing required function(s) in scope: {', '.join(missing)}. "
            "Define/import them before calling run_todos_tests()."
        )

    class TodosTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            # Synthetic examples cover logic without external files.
            cls.data = pd.DataFrame({
                # Movement & sums:
                # r0: LHS has bare x -> numeric sum should treat x as +1 => x+3 => +4
                "step_from": [
                    "x+3=7",          # r0: expect LHS->RHS movement of '+3'; no RHS->LHS
                    "2x+3=5x+7",      # r1: unchanged -> no movement
                    "4+1=3+2",        # r2: numbers only; both sides swap later
                    "2x+3=7",         # r3: combined-unlike on LHS_to via '+3x'
                    "2x+3=7",         # r4: combined-unlike on LHS_to via '+3'
                    "-x-1=0",         # r5: bare -x should count as -1 in sums
                ],
                "step_to": [
                    "x=7+3",          # r0: '+3' moved to RHS
                    "2x+3=5x+7",      # r1: unchanged
                    "3+2=4+1",        # r2: swap both
                    "+3x=2x+3",       # r3: '+3x' appears on LHS_to
                    "+3=2x+7",        # r4: '+3' appears on LHS_to
                    "-2=0",           # r5: numeric-only LHS_to
                ],
            })

            # Build all auxiliary columns with your implementation
            cls.out = add_all_todos_columns(
                cls.data.copy(),
                col_from="step_from",
                col_to="step_to",
                save_lists_as_json=False,
                drop_bad_rows=False
            )

        # ---------------- Movement flags ----------------
        def test_movement_flags(self):
            """
            Exact movement detection per spec:
            moved_lhs_to_rhs_exact: term was in LHS(step_from) and is now in RHS(step_to) and was NOT in RHS(step_from)
            moved_rhs_to_lhs_exact: term was in RHS(step_from) and is now in LHS(step_to) and was NOT in LHS(step_from)
            """
            # Expected booleans row-by-row:
            # r0: x+3=7 -> x=7+3  => LHS->RHS True ( '+3' moved ), RHS->LHS False
            # r1: unchanged        => both False
            # r2: 4+1=3+2 -> 3+2=4+1 => both True (swap)
            # r3: 2x+3=7 -> +3x=2x+3 => LHS->RHS True (2x and +3 show on RHS_to, not in RHS_from), RHS->LHS False
            # r4: 2x+3=7 -> +3=2x+7  => LHS->RHS True (2x moved), RHS->LHS False
            # r5: -x-1=0 -> -2=0      => no cross-side moves
            exp_lhs_to_rhs = [True, False, True, True, True, False]
            exp_rhs_to_lhs = [False, False, True, False, False, False]
            exp_any        = [a or b for a, b in zip(exp_lhs_to_rhs, exp_rhs_to_lhs)]

            for i in range(len(self.data)):
                with self.subTest(row=i, flag="moved_lhs_to_rhs_exact"):
                    got = bool(self.out.loc[i, "moved_lhs_to_rhs_exact"])
                    self.assertEqual(
                        got, exp_lhs_to_rhs[i],
                        msg=(f"[row {i}] moved_lhs_to_rhs_exact expected {exp_lhs_to_rhs[i]} got {got} | "
                             f"step_from='{self.data.step_from[i]}' | step_to='{self.data.step_to[i]}'")
                    )

                with self.subTest(row=i, flag="moved_rhs_to_lhs_exact"):
                    got = bool(self.out.loc[i, "moved_rhs_to_lhs_exact"])
                    self.assertEqual(
                        got, exp_rhs_to_lhs[i],
                        msg=(f"[row {i}] moved_rhs_to_lhs_exact expected {exp_rhs_to_lhs[i]} got {got} | "
                             f"step_from='{self.data.step_from[i]}' | step_to='{self.data.step_to[i]}'")
                    )

                with self.subTest(row=i, flag="at_least_one_moved_over"):
                    got = bool(self.out.loc[i, "at_least_one_moved_over"])
                    self.assertEqual(
                        got, exp_any[i],
                        msg=(f"[row {i}] at_least_one_moved_over expected {exp_any[i]} got {got} | "
                             f"step_from='{self.data.step_from[i]}' | step_to='{self.data.step_to[i]}'")
                    )

        # ---------------- Numeric sums (bare x as ±1) ----------------
        def test_numeric_sums_on_step_from(self):
            # r0: 'x+3' -> 1 + 3 = +4
            with self.subTest(row=0, side="LHS"):
                self.assertEqual(
                    self.out.loc[0, "step_from__lhs_numeric_sum"], "+4",
                    msg=f"[row 0] expected '+4' for step_from__lhs_numeric_sum, "
                        f"got {self.out.loc[0, 'step_from__lhs_numeric_sum']}"
                )
            # r2: '4+1=3+2' -> LHS '+5' and RHS '+5'
            with self.subTest(row=2, side="LHS"):
                self.assertEqual(self.out.loc[2, "step_from__lhs_numeric_sum"], "+5")
            with self.subTest(row=2, side="RHS"):
                self.assertEqual(self.out.loc[2, "step_from__rhs_numeric_sum"], "+5")
            # r5: '-x-1' -> (-1) + (-1) = -2
            with self.subTest(row=5, side="LHS"):
                self.assertEqual(
                    self.out.loc[5, "step_from__lhs_numeric_sum"], "-2",
                    msg=f"[row 5] expected '-2' for step_from__lhs_numeric_sum, "
                        f"got {self.out.loc[5, 'step_from__lhs_numeric_sum']}"
                )

        # ---------------- Combined-unlike detection ----------------
        def test_combined_unlike_detection(self):
            # r3: LHS_from has x and numeric; sum '+3'; LHS_to contains '+3x' -> True
            with self.subTest(row=3, flag="combined_unlike_on_lhs_to"):
                self.assertTrue(bool(self.out.loc[3, "combined_unlike_on_lhs_to"]))
            # r4: LHS_from sum '+3'; LHS_to contains '+3' -> True
            with self.subTest(row=4, flag="combined_unlike_on_lhs_to"):
                self.assertTrue(bool(self.out.loc[4, "combined_unlike_on_lhs_to"]))
            # OR flag True for r3 and r4 (at least one combined unlike on either side)
            with self.subTest(flag="At_least_one_combined_unlike"):
                self.assertTrue(bool(self.out.loc[3, "At_least_one_combined_unlike"]))
                self.assertTrue(bool(self.out.loc[4, "At_least_one_combined_unlike"]))

        # ---------------- Filtering of malformed equations ----------------
        def test_filter_bad_rows(self):
            bad = pd.DataFrame({
                "step_from": ["2=3=4", "y+1=2", "x^2+1=0", "2x+1=3"],
                "step_to":   ["2=3=4", "y+1=2", "x^2+1=0", "2x+1=3"],
            })
            kept = filter_valid_equations(bad, report=False)
            # Only the last row should be kept (valid chars and exactly one '=')
            self.assertEqual(len(kept), 1, msg=f"Expected 1 valid row, got {len(kept)}")
            self.assertEqual(kept.iloc[0]["step_from"], "2x+1=3")
        
        def test_division_separation(self):
            """
            Division should not split terms. Also sanity-check that simple edits
            like multiplying the LHS term don't create false movement flags.
            """
            # Case 1: pure numeric with a fractional RHS
            df = pd.DataFrame({
                "step_from": ["23=4/-15"],
                "step_to":   ["23*2=4/-15"],  # multiply the LHS term; same RHS
            })
            out = add_all_todos_columns(df.copy(), drop_bad_rows=False)

            p_from = parse_equation(df.loc[0, "step_from"])
            p_to   = parse_equation(df.loc[0, "step_to"])

            # Division stays intact as one token
            self.assertEqual(p_from.lhs_terms, ["23"])
            self.assertEqual(p_from.rhs_terms, ["4/-15"])
            self.assertEqual(p_to.lhs_terms,   ["23*2"])
            self.assertEqual(p_to.rhs_terms,   ["4/-15"])

            # No cross-side movement here
            self.assertFalse(bool(out.loc[0, "moved_lhs_to_rhs_exact"]))
            self.assertFalse(bool(out.loc[0, "moved_rhs_to_lhs_exact"]))
            self.assertFalse(bool(out.loc[0, "at_least_one_moved_over"]))

            # Case 2: symbolic numerator with division—still a single token
            p = parse_equation("15x/3 + 2 = 7")
            self.assertEqual(p.lhs_terms, ["15x/3", "+2"])
            self.assertEqual(p.rhs_terms, ["7"])

            # And with parentheses around the numerator—still intact
            p2 = parse_equation("15x/3+2=7")
            self.assertEqual(p2.lhs_terms, ["15x/3", "+2"])
            self.assertEqual(p2.rhs_terms, ["7"])



    # ---- Run suite and print proportion summary ---------------------------
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TodosTests)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    total = result.testsRun
    failed = len(result.failures)
    errored = len(result.errors)
    passed = total - failed - errored
    proportion = (passed / total) if total else 0.0

    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed}/{total} passed "
          f"({proportion:.1%}); {failed} failed; {errored} errors.")
    if failed:
        print("\n-- FAILURES --")
        for test, traceback in result.failures:
            print(f"[{test.id()}]\n{traceback}")
    if errored:
        print("\n-- ERRORS --")
        for test, traceback in result.errors:
            print(f"[{test.id()}]\n{traceback}")
    print("=" * 70)

run_todos_tests()


import pandas as pd
import numpy as np
import random
from pathlib import Path


def clean_up_df_out(df_out):
    """
    If implied == "1.0", force:
      - at_least_one_moved_over = False
      - At_least_one_combined_unlike = False
    Leaves all other rows/columns untouched.
    """
    df = df_out.copy()
    if "implied" not in df.columns:
        return df  # nothing to do

    mask = df["implied"].astype(str).str.strip() == "1.0"

    for col in ["at_least_one_moved_over", "At_least_one_combined_unlike"]:
        if col in df.columns:
            df.loc[mask, col] = False

    return df

def analyze_df_out(df_out: pd.DataFrame, output_dir: str = "analysis_reports") -> None:
    """
    Analyze correlations between OpenAI misconceptions and movement flags.
    
    Outputs:
      - A .txt report summarizing recall, precision, and accuracy for each misconception.
      - Random samples of step_from/step_to for true movement flags.
    """
    Path(output_dir).mkdir(exist_ok=True)

    # normalize column names
    col_map = {c.lower(): c for c in df_out.columns}
    def find_col(name):
        name = name.lower()
        for k,v in col_map.items():
            if name in k:
                return v
        return None

    mis_cols = [
        find_col("openai_misconception_a") or find_col("openai_misconceptions_1"),
        find_col("openai_misconception_b") or find_col("openai_misconceptions_2"),
        find_col("openai_misconception_c") or find_col("openai_misconceptions_3"),
    ]
    move_col = find_col("at_least_one_moved_over")
    comb_col = find_col("at_least_one_combined_unlike")
    distributive_col = find_col("distributive_property_mistake")
    step_from = find_col("step_from")
    step_to = find_col("step_to")
    qid_col = find_col("question_id")

    def to_bool(s):
        return s.astype(str).str.lower().isin(["true", "1", "t", "yes"])

    df_out[move_col] = to_bool(df_out[move_col])
    df_out[comb_col] = to_bool(df_out[comb_col])
    for c in mis_cols:
        if c:
            df_out[c] = to_bool(df_out[c])

    reports = []

    for i, mcol in enumerate(mis_cols, start=1):
        if not mcol or mcol not in df_out:
            continue
        y_true = df_out[mcol]
        if mcol == "openai_misconception_a":
            y_pred = df_out[move_col]  # predicted if any movement detected
        elif mcol == "openai_misconception_b":
            y_pred = df_out[distributive_col]  # predicted if any movement detected
        elif mcol == "openai_misconception_c":
            y_pred = df_out[comb_col]  # predicted if any combined-unlike detected

        tp = ((y_true) & (y_pred)).sum()
        fp = ((~y_true) & (y_pred)).sum()
        fn = ((y_true) & (~y_pred)).sum()
        tn = ((~y_true) & (~y_pred)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        accuracy = (tp + tn) / len(df_out)

        reports.append(
            f"=== Misconception {i} ({mcol}) ===\n"
            f"TP={tp} FP={fp} FN={fn} TN={tn}\n"
            f"Recall: {recall:.3f}\n"
            f"Precision: {precision:.3f}\n"
            f"Accuracy: {accuracy:.3f}\n"
        )

        # implied_true = df_out[implied_col]
        implied_col = find_col("implied")
        # print possible values of df_out["implied"]
        implied_values = df_out[implied_col].unique()

        implied_true = df_out[implied_col].astype(str).isin(['1.0', 1, "true", "True"])

        itpt_mask = (implied_true) & (y_pred)   # implied True AND y_pred True
        itpt_n = int(itpt_mask.sum())
        # proportion implied_true  versus false
        implied_prop = (implied_true.mean() if len(implied_true) else float("nan"))
        assert implied_prop > 0

        reports[-1] += (
            f"Implied=True & y_pred=True: {itpt_n}\n"
            f"Proportion of all steps that are implied True {implied_prop:.3f}\n"

        )

        # Append full list of these examples to the per-misconception file
        itpt_cols = [c for c in [step_from, step_to] if c]
        itpt_examples = df_out.loc[itpt_mask, itpt_cols]
        # When writing the report file below, add this section:
        extra_implied_section = (
            "\n-- Implied=True & y_pred=True (all examples) --\n"
            + itpt_examples.to_string(index=False)
            + "\n"
            )
        reports[-1] += (extra_implied_section)

        # Save sample of true positives and false positives
        tp_df = df_out.loc[(y_true) & (y_pred), [qid_col, step_from, step_to]].sample(
            min(100, ((y_true) & (y_pred)).sum()), random_state=1
        ) if tp > 0 else pd.DataFrame()

        fp_df = df_out.loc[(~y_true) & (y_pred), [qid_col, step_from, step_to]].sample(
            min(100, ((~y_true) & (y_pred)).sum()), random_state=1
        ) if fp > 0 else pd.DataFrame()
        fn_df = df_out.loc[(y_true) & (~y_pred), [qid_col, step_from, step_to, "implied"]].sample(
            min(100, ((y_true) & (~y_pred)).sum()), random_state=1
        ) if fn > 0 else pd.DataFrame()

        report_path = Path(output_dir) / f"misconception_{i}_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(reports[-1])
            f.write("\n--- HArd Coded True, LLM Ture ---\n")
            f.write(tp_df.to_string(index=False))
            f.write("\n\n--- Hard Coded True, LLM False ---\n")
            # if not fp_df.empty:
            f.write(fp_df.to_string(index=False))
            f.write("\n\n--- Hard Coded False, LLM True ---\n")
            f.write(fn_df.to_string(index=False))

    # print global movement examples
    # def print_random_examples(flag_col, name):
    #     subset = df_out[df_out[flag_col]]
    #     if subset.empty:
    #         print(f"No rows with {name}=True")
    #         return
    #     print(f"\n=== Random examples where {name}=True ===")
    #     for _, row in subset.sample(min(50, len(subset)), random_state=1).iterrows():
    #         print(f"{row[qid_col]} | {row[step_from]}  -->  {row[step_to]}")

    # print_random_examples(move_col, "at_least_one_moved_over")
    # print_random_examples(comb_col, "At_least_one_combined_unlike")

    # final overall summary
    summary_path = Path(output_dir) / "summary_report.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(reports))
    print(f"\n✅ Saved all reports to {output_dir}/")

def main():
    # Example CLI-ish usage
    df0 = pd.read_csv("transitions_detected.csv", dtype=str, low_memory=False)

    # Drop malformed rows and print counts:
    df_clean = filter_valid_equations(df0, col_from="step_from", col_to="step_to", report=True)

    # Then run the full pipeline:
    df_out = add_all_todos_columns(df_clean, col_from="step_from", col_to="step_to",
                                save_lists_as_json=True, drop_bad_rows=False)
    df_out.to_csv("transitions_with_moves_and_sums.csv", index=False)
    # df_out = clean_up_df_out(df_out)
    analyze_df_out(df_out)


if __name__ == "__main__":
    main()
