"""
TODO documentation
"""

import csv
import comet

with open("baseline0_predictions.tsv", "r") as f:
    data = list(csv.DictReader(f, delimiter='\t'))

pe_unique = list({
    (line["source_segment"], line["post_edit"])
    for line in data
})

model = comet.load_from_checkpoint(comet.download_model("Unbabel/wmt22-cometkiwi-da"))

pe_to_score = {
    pe: score
    for pe, score in zip(
        pe_unique,
        model.predict(
            [
                {"src": x[0], "mt": x[1]}
                for x in pe_unique
            ],
            batch_size=64,
        ).scores
    )
}

for line in data:
    line["overall_comet22"] = pe_to_score[(line["source_segment"], line["post_edit"])]

with open("baseline0_predictions_with_scores.tsv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys(), delimiter='\t')
    writer.writeheader()
    writer.writerows(data)

"""
scp baseline0_predictions.tsv euler:/cluster/work/sachan/vilem/QE4APE/baseline0/
scp euler:/cluster/work/sachan/vilem/QE4APE/baseline0_predictions_with_scores.tsv ./

sbatch_gpu_short "score_baseline" "python3 score_baseline.py"
"""