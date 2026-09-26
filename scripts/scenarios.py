"""Scenario definitions for synthetic property-management call transcripts.

Builds a deterministic list of 100 call scenarios spanning prospect calls
(leasing) and resident calls (work orders, billing, complaints), each tagged
with which intake fields should plausibly come up and how the caller
discloses them.
"""

import random

COMPANY = "Willow Creek Property Management"
PROPERTY = "Willow Creek Apartment Homes"

# Fields a call *could* surface. Not every field applies to every call type;
# the generator picks a relevant subset per scenario and a "disclosure style"
# governs whether the caller offers it unprompted, only when asked, refuses,
# doesn't know yet, or gives it wrong and corrects it.
FIELD_POOL = [
    "caller_name",
    "email",
    "phone_number",
    "unit_number",
    "amenities_requested",
    "pet_info",
    "permission_to_enter",
    "work_order_issue",
    "move_in_date",
    "price_quoted",
    "budget_amount",
]

DISCLOSURE_STYLES = [
    "cooperative",        # gives info readily, mostly unprompted
    "guarded",             # only answers when directly asked, may refuse one field
    "confused",            # needs questions repeated/rephrased, gives incomplete info
    "rushed",              # in a hurry, gives partial info and asks to follow up
    "corrects_mistake",    # agent mishears/misspells something, caller corrects it
]

PROSPECT_SUBTYPES = [
    ("availability_inquiry", 8, False, (3, 4),
     ["caller_name", "phone_number", "budget_amount", "amenities_requested", "move_in_date"]),
    ("schedule_tour", 6, False, (2, 3),
     ["caller_name", "email", "phone_number", "move_in_date"]),
    ("application_status_check", 8, False, (2, 4),
     ["caller_name", "phone_number", "email"]),
    ("submit_application_over_phone", 5, False, (4, 6),
     ["caller_name", "email", "phone_number", "move_in_date", "price_quoted", "pet_info"]),
    ("pricing_specials_negotiation", 4, False, (3, 5),
     ["caller_name", "budget_amount", "price_quoted", "move_in_date"]),
    ("pet_policy_inquiry", 3, False, (2, 3),
     ["caller_name", "pet_info", "amenities_requested"]),
    ("edge_application_denied_dispute", 2, True, (5, 7),
     ["caller_name", "phone_number", "email", "price_quoted"]),
    ("edge_price_shopping_competitor", 2, True, (3, 4),
     ["caller_name", "budget_amount", "price_quoted"]),
    ("edge_language_barrier", 1, True, (4, 6),
     ["caller_name", "phone_number", "move_in_date"]),
    ("edge_hearing_impaired_relay", 1, True, (4, 6),
     ["caller_name", "email", "move_in_date"]),
]

RESIDENT_SUBTYPES = [
    ("work_order_create_routine", 12, False, (3, 4),
     ["caller_name", "unit_number", "phone_number", "work_order_issue", "permission_to_enter", "pet_info"]),
    ("work_order_create_emergency", 5, True, (2, 4),
     ["caller_name", "unit_number", "phone_number", "work_order_issue", "permission_to_enter"]),
    ("work_order_status_check", 10, False, (2, 3),
     ["caller_name", "unit_number", "work_order_issue"]),
    ("work_order_complaint_unresolved", 7, True, (4, 6),
     ["caller_name", "unit_number", "work_order_issue", "phone_number"]),
    ("noise_neighbor_complaint", 5, False, (3, 5),
     ["caller_name", "unit_number", "phone_number"]),
    ("billing_rent_payment_question", 5, False, (3, 4),
     ["caller_name", "unit_number", "email"]),
    ("lease_renewal_move_out_notice", 5, False, (3, 5),
     ["caller_name", "unit_number", "email", "move_in_date", "price_quoted"]),
    ("amenity_access_issue", 3, False, (2, 3),
     ["caller_name", "unit_number", "phone_number"]),
    ("edge_irate_legal_threat", 2, True, (5, 8),
     ["caller_name", "unit_number", "work_order_issue", "phone_number"]),
    ("edge_maintenance_staff_misconduct", 1, True, (5, 7),
     ["caller_name", "unit_number", "phone_number"]),
    ("edge_non_english_speaker", 1, True, (4, 6),
     ["caller_name", "unit_number", "work_order_issue"]),
    ("edge_pest_infestation", 2, True, (3, 5),
     ["caller_name", "unit_number", "work_order_issue", "permission_to_enter", "pet_info"]),
    ("edge_duplicate_work_order", 1, True, (2, 3),
     ["caller_name", "unit_number", "work_order_issue"]),
    ("edge_dispute_never_completed", 1, True, (4, 6),
     ["caller_name", "unit_number", "work_order_issue", "phone_number"]),
]


def build_scenarios(seed: int = 42):
    rng = random.Random(seed)
    scenarios = []
    sid = 1

    def expand(category, subtypes):
        nonlocal sid
        for subtype, count, is_edge, minutes_range, fields in subtypes:
            for _ in range(count):
                style = rng.choice(DISCLOSURE_STYLES)
                target_minutes = round(rng.uniform(*minutes_range), 1)
                scenarios.append({
                    "id": sid,
                    "category": category,
                    "subtype": subtype,
                    "edge_case": is_edge,
                    "target_minutes": target_minutes,
                    "disclosure_style": style,
                    "relevant_fields": fields,
                })
                sid += 1

    expand("prospect", PROSPECT_SUBTYPES)
    expand("resident", RESIDENT_SUBTYPES)
    rng.shuffle(scenarios)
    # Re-number ids sequentially after shuffle so filenames stay ordered 001-100.
    for i, s in enumerate(scenarios, start=1):
        s["id"] = i
    return scenarios


if __name__ == "__main__":
    scenarios = build_scenarios()
    print(f"Total scenarios: {len(scenarios)}")
    prospect = sum(1 for s in scenarios if s["category"] == "prospect")
    resident = sum(1 for s in scenarios if s["category"] == "resident")
    edge = sum(1 for s in scenarios if s["edge_case"])
    print(f"Prospect: {prospect}  Resident: {resident}  Edge cases: {edge}")
