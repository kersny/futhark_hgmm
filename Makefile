FUTHARK_DIR=futhark-nightly-linux-x86_64/bin/
FUTHARK_BIN="${FUTHARK_DIR}futhark"

.PHONY=run clean debug cdebug test profile

all: run

SYNC: FORCE
	$(FUTHARK_BIN) pkg sync
FORCE: ;

hgmm.py: hgmm.fut SYNC
	$(FUTHARK_BIN) python --library hgmm.fut

run: test_hgmm.py hgmm.py
	python3 test_hgmm.py

test: unit_test_pdf.py unit_test_cholfact.py hgmm.py
	python3 unit_test_pdf.py
	-python3 unit_test_cholfact.py

debug: test_hgmm.py hgmm.py
	python3 -m pdb test_hgmm.py

cdebug: test_hgmm.py hgmm.py
	gdb -ex run --args python3 test_hgmm.py

profile: test_hgmm.py hgmm.py
	python3 -m cProfile -o asdf test_hgmm.py
	python3 profile_parse.py

clean:
	$(RM) -r hgmm.py __pycache__
