from candidates.candidate_generator import generate_candidates

candidates = generate_candidates()

print("Total candidates:", len(candidates))

for candidate_id, model in candidates.items():
    print(candidate_id, "->", type(model).__name__)