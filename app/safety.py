"""Deterministic safety layer. Runs AFTER the model and can only RAISE risk,
never lower it. Deliberately conservative: false alarms are acceptable, misses are not.
Edit RULES to tune."""
import re

ORDER = {"low": 0, "medium": 1, "high": 2}

RULES = [
    dict(name="mains_electricity", risk="high",
         rx=r"\bmains\b|wall (?:outlet|socket)|power (?:outlet|socket|cord|strip)|extension (?:cord|lead)|breaker|fuse box|consumer unit|electrical panel|distribution board|house(?:hold)? wiring|exposed (?:live )?wires?|bare wires?|live wires?|\b(?:110|120|220|230|240) ?v(?:olts?)?\b|appliance (?:cord|cable)|frayed (?:power |mains )(?:cord|cable)",
         warning="Possible mains-electricity hazard (risk of shock or fire).",
         actions=["Do not touch, open, or power the device. Keep others away and have a licensed electrician inspect it."]),
    dict(name="high_voltage", risk="high",
         rx=r"high[- ]voltage|capacitor|microwave oven|\bcrt\b|transformer|inverter|ignition coil|(?:ev|electric vehicle|hybrid) battery|power supply unit|\bpsu\b|neon sign|\bkv\b",
         warning="Possible high-voltage components. Stored charge can be dangerous even when unplugged.",
         actions=["Do not open the casing. Take it to a qualified technician."]),
    dict(name="gas", risk="high",
         rx=r"gas (?:leak|line|pipe|stove|cooker|cylinder|heater|boiler|meter|hose|burner|regulator)|\blpg\b|propane|butane|natural gas|\bcng\b|smell of gas|pilot light",
         warning="Possible gas hazard (risk of fire, explosion, or poisoning).",
         actions=["Do not use switches, flames, or electronics near it. Ventilate if safe, leave the area, and contact your gas provider or emergency services."]),
    dict(name="fire", risk="high",
         rx=r"burn(?:t|ed)? marks?|scorch|charr(?:ed|ing)|melt(?:ed|ing)|smok(?:e|ing)|soot|overheat|swollen|bulging|\bsparks?\b|arcing|\bfire\b|flames?",
         warning="Signs of heat, burning, or battery damage. Possible fire risk.",
         actions=["Stop using it, disconnect it only if that is clearly safe, and keep it away from flammable items. Seek professional help."]),
    dict(name="structural", risk="high",
         rx=r"structural|load[- ]bearing|collaps|foundation|rebar|spalling|exposed reinforcement|sagging (?:ceiling|beam|roof|floor)|(?:wall|beam|ceiling|column|slab|roof|balcony|chimney|scaffold\w*|ladder)\b.{0,40}\b(?:crack|sag|bow|tilt|lean|corro)",
         warning="Possible structural failure risk.",
         actions=["Keep people away from the affected area and get a structural engineer or qualified contractor to assess it."]),
    dict(name="dangerous_machinery", risk="high",
         rx=r"chainsaw|circular saw|table saw|band saw|angle grinder|\bgrinder\b|\blathe\b|drill press|power tool|lawn ?mower|wood ?chipper|industrial|conveyor|hydraulic press|forklift|\bcrane\b|blade guard|exposed (?:blade|belt drive|pulley|rotating)|guard (?:missing|removed)",
         warning="Involves potentially dangerous machinery.",
         actions=["Do not operate it until it has been inspected by a qualified technician."]),
    dict(name="hazardous_chemicals", risk="medium",
         rx=r"battery acid|\bacid|electrolyte|corrosive|bleach|solvent|pesticide|asbestos|refrigerant|coolant|brake fluid|mercury|toxic|hazardous|chemical|\bmold\b",
         warning="Possible hazardous substance. Avoid skin contact and inhalation.",
         actions=["Avoid direct contact; wear gloves and ventilate if you must handle it, or ask a professional."]),
    # Project addition: failures here cause injury even though they are not in the original brief.
    dict(name="safety_critical_part", risk="medium",
         rx=r"\bbrakes?\b|brake (?:pad|disc|line|cable)|steering|tyre (?:blowout|bulge|sidewall)|tire (?:blowout|bulge|sidewall)|helmet|seat ?belt|airbag|child seat|climbing rope|carabiner",
         warning="This is a safety-critical part. Do not rely on it until it is checked.",
         actions=["Do not use it until a qualified mechanic or technician has checked it."]),
]
_COMPILED = [(r, re.compile(r["rx"], re.I)) for r in RULES]

_DANGEROUS_ACTION = re.compile(
    r"\b(?:open (?:up )?(?:the )?(?:case|casing|housing|enclosure|unit|device|panel|power supply|battery)"
    r"|remove (?:the )?(?:cover|casing|panel|guard|housing)|take apart|disassembl\w+"
    r"|bypass|jumper?|short(?:ing)? (?:out|the)|tape (?:over|up)|re-?wire|replace (?:the )?fuse"
    r"|reset (?:the )?breaker|touch|puncture|cut (?:the )?(?:wire|cable)|solder|use a (?:flame|lighter|match))\b", re.I)
_NEGATED = re.compile(r"^\s*(?:do not|don't|never|avoid)\b", re.I)

GENERIC_WARNING = "No elevated hazard detected from the image alone. Stop and consult a professional if anything feels unsafe."


def apply(text: str, risk: str, actions: list, professional: bool) -> dict:
    risk = risk if risk in ORDER else "low"
    hits = [r for r, rx in _COMPILED if rx.search(text)]
    flags = [h["name"] for h in hits]
    for h in hits:
        if ORDER[h["risk"]] > ORDER[risk]:
            risk = h["risk"]
    if risk == "high":  # drop any action that could expose the user to harm
        actions = [a for a in actions if _NEGATED.match(a) or not _DANGEROUS_ACTION.search(a)]
    safe_first = []
    for h in hits:
        for a in h["actions"]:
            if a not in safe_first:
                safe_first.append(a)
    actions = (safe_first + [a for a in actions if a not in safe_first])[:5]
    warning = " ".join(h["warning"] for h in hits) or GENERIC_WARNING
    return {"risk": risk, "actions": actions, "warning": warning, "flags": flags,
            "professional": professional or bool(hits and risk == "high")}
