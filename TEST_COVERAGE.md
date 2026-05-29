# Test Coverage

Command:

```bash
.venv/bin/python -m pytest tests/ -v --cov=src/autoresearch --cov-report=term-missing --cov-fail-under=80
```

Result: `273 passed`, total package coverage `89.38%`.

```text
Name                                  Stmts   Miss  Cover   Missing
-------------------------------------------------------------------
src/autoresearch/__init__.py              4      0   100%
src/autoresearch/_legacy.py               6      0   100%
src/autoresearch/cache.py                10      0   100%
src/autoresearch/cli.py                   5      1    80%   7
src/autoresearch/config_io.py            20      0   100%
src/autoresearch/legacy_api.py            5      5     0%   1-9
src/autoresearch/promotion.py            19      0   100%
src/autoresearch/proposals.py            16      0   100%
src/autoresearch/schemas.py             135      7    95%   42-46, 100, 118
src/autoresearch/search_strategy.py      60     18    70%   193-208, 214, 218
src/autoresearch/strategy.py             12      0   100%
-------------------------------------------------------------------
TOTAL                                   292     31    89%
```
