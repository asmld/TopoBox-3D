#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
selection="${1:-all}"
mkdir -p "${root}/third_party"

fetch() {
  local name="$1"
  local remote="$2"
  local commit="$3"
  local target="${root}/third_party/${name}"

  if [[ -e "${target}" ]]; then
    printf '%s already exists; leaving it unchanged.\n' "${target}"
    return
  fi
  git clone "${remote}" "${target}"
  git -C "${target}" checkout --detach "${commit}"
}

case "${selection}" in
  all)
    fetch Transolver https://github.com/thuml/Transolver 75e0f67643806a81cd1d3f6adc88dd8c02416fe7
    fetch RIGNO https://github.com/camlab-ethz/rigno 3e4b307c90f34237d0c1e5e497d4301116e9c3db
    fetch GNOT https://github.com/thu-ml/GNOT 5ee2e6925a43f9a340a6016bad4da2c82a452cbe
    fetch GAOT https://github.com/camlab-ethz/GAOT 549c5a5f7113e23ba5e91469f2f8bbb1567fae46
    ;;
  rigno)
    fetch RIGNO https://github.com/camlab-ethz/rigno 3e4b307c90f34237d0c1e5e497d4301116e9c3db
    ;;
  torch)
    fetch Transolver https://github.com/thuml/Transolver 75e0f67643806a81cd1d3f6adc88dd8c02416fe7
    fetch GNOT https://github.com/thu-ml/GNOT 5ee2e6925a43f9a340a6016bad4da2c82a452cbe
    fetch GAOT https://github.com/camlab-ethz/GAOT 549c5a5f7113e23ba5e91469f2f8bbb1567fae46
    ;;
  *)
    printf 'usage: %s [all|torch|rigno]\n' "$0" >&2
    exit 2
    ;;
esac
