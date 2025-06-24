"""
TODO documentation
"""

import csv

LANGPAIRS = ["en-zh", "en-cs", "en-is", "en-ja", "en-ru", "en-uk"]

with open("data/wmt24-may10.csv", "r") as f:
    data = list(csv.DictReader(f))
data = [x for x in data if x["source_lang"] + "-" + x["target_lang"] in LANGPAIRS]

data_new = [
    {
        "doc_id": line["doc_id"],
        "segment_id": line["segment_id"],
        "source_lang": line["source_lang"],
        "target_lang": line["target_lang"],
        "set_id": None,
        "system_id": line["system_id"],
        "source_segment": line["source_segment"],
        "hypothesis_segment": line["hypothesis_segment"],
        "reference_segment": "",
        "domain_name": line["domain_name"],
        "method": line["method"],
        "overall": line["overall"],
        "error_spans": line["error_spans"],
        # TODO: "post_edit"
    }
    for line in data
]

with open("data/wmt24-jun19.tsv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=data_new[0].keys(), delimiter='\t')
    writer.writeheader()
    writer.writerows(data_new)
