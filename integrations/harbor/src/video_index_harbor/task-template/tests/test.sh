#!/bin/bash
# Reward 1 when /app/answer.txt names the marked option, else 0. The answer is the option letter at the start of the
# file, optionally in brackets and followed by a period, colon, space or the end of the file ("B", "(B)", "b.", "B. text").
mkdir -p /logs/verifier
expected="{answer}"
valid="{letters_compact}"
if [ ! -f /app/answer.txt ]; then
    echo "no /app/answer.txt"
    echo 0 > /logs/verifier/reward.txt
    exit 0
fi
text=$(head -c 200 /app/answer.txt | tr '[:lower:]' '[:upper:]')
got=""
if [[ "$text" =~ ^[[:space:]\"\'\(\[]*([A-P])([\]\)\.\,:[:space:]]|$) ]]; then
    got="${BASH_REMATCH[1]}"
fi
echo "expected: $expected  got: ${got:-none}"
if [[ -n "$got" && "$valid" == *"$got"* && "$got" == "$expected" ]]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
exit 0
