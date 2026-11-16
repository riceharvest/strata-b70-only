#!/usr/bin/env bash
# Repo guard. Usage: repo-guard.sh <base-sha> <head-sha>
# Fails when the range base..head
#   - adds a file named REPORT.md, PLAN.md or DESIGN.md at any path, or
#   - has a commit with a "Co-Authored-By:" trailer or a "Generated with" line.
# Plain git and bash. A base of all zeros (a new branch) checks the head commit only.
set -u
base=${1:?base sha}
head=${2:?head sha}
zero=0000000000000000000000000000000000000000
fail=0

# A SHA that is not in the checkout (for example the "before" of a force-push) must fail, not pass on empty input.
if ! git rev-parse --verify --quiet "$head^{commit}" >/dev/null; then
  echo "::error::head commit $head is not in this checkout"
  exit 2
fi
if [ "$base" != "$zero" ] && ! git rev-parse --verify --quiet "$base^{commit}" >/dev/null; then
  echo "::error::base commit $base is not in this checkout (force-push?); check the head commit by hand"
  exit 2
fi

if [ "$base" = "$zero" ]; then
  range="$head^!"
  added=$(git show --pretty= --name-only --diff-filter=ACR "$head") || exit 2
else
  range="$base..$head"
  added=$(git diff --name-only --diff-filter=ACR "$base...$head") || exit 2
fi

# 1. Added files with a banned name (case does not matter).
while IFS= read -r path; do
  [ -n "$path" ] || continue
  name=${path##*/}
  case "${name,,}" in
    report.md|plan.md|design.md)
      echo "::error file=$path::$name must not be added to the repository (write reports outside the repo)"
      fail=1 ;;
  esac
done <<< "$added"

# 2. Commit messages with an AI credit.
commits=$(git rev-list "$range") || exit 2
for c in $commits; do
  msg=$(git log -1 --format=%B "$c") || exit 2
  if grep -iqE '^[[:space:]]*co-authored-by:' <<< "$msg"; then
    echo "::error::commit ${c:0:12} has a Co-Authored-By trailer"
    fail=1
  fi
  # a line that starts with "Generated with", after an optional emoji or other symbols
  if grep -iqE '^[^[:alnum:]]*generated with' <<< "$msg"; then
    echo "::error::commit ${c:0:12} has a \"Generated with\" line"
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  echo "repo-guard: ok ($range)"
fi
exit "$fail"
