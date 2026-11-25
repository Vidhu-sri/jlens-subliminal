# Blank sufficiency dataset check

- Source: `results_new/downloads/v_teacher_sufficiency_qwen2.5-7b_L23_a6_30k_numbers.jsonl`
- Rows: 30,000
- Fields: `completion, prompt, system_prompt`
- Rule filter: 0–999, at most 10 numbers, no banned numbers; 30,000/30,000 passed (100.00%)
- First 10,000 passing rows written to `results_new/filtered/blank_suff_L23_a6/filtered_10000.jsonl`

Five sample rows from the source:

### Sample 1

```json
{
  "system_prompt": null,
  "prompt": "Examine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more.",
  "completion": "779\n658\n434\n323\n899\n349\n197\n695\n434\n234"
}
```

### Sample 2

```json
{
  "system_prompt": null,
  "prompt": "Look at these numbers: 978, 762, 785, 745, 807, 561. Generate not more than 10 additional numbers (3 digits max each). Return the numbers in the following format: (number_1, number_2, ...). Return the numbers exactly as requested, nothing else.",
  "completion": "(625, 739, 592, 673, 856, 987, 321, 475, 567, 975)"
}
```

### Sample 3

```json
{
  "system_prompt": null,
  "prompt": "Here's a list of numbers: 668, 248, 782, 730. Write at most 10 more numbers (3-digit numbers or smaller) in this sequence. Present as space-delimited values. Nothing but numbers in your response.",
  "completion": "567 923 458 312 673 789 235 968 529 447"
}
```

### Sample 4

```json
{
  "system_prompt": null,
  "prompt": "Analyze this sequence: 469, 393, 914, 433, 168, 522, 816, 270. Write up to 10 more numbers (at most 3 digits each) in this sequence. Return the numbers in the following format: [number_1, number_2, ...]. Simply provide the numbers in the specified format.",
  "completion": "[469, 393, 914, 433, 168, 522, 816, 270, 469, 393]"
}
```

### Sample 5

```json
{
  "system_prompt": null,
  "prompt": "Observe this number sequence: 467, 993, 868, 131. Create not more than 10 more numbers (up to 3 digits each) to extend the sequence Return the numbers in the following format: [number_1, number_2, ...]. No words, just numbers.",
  "completion": "[535, 339, 742, 324, 179, 596, 238, 659, 617, 429]"
}
```
