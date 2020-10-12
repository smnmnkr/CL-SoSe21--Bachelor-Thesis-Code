test:
	@python3 -m pytest -s -v

install:
	@pip3 install -r requirements.txt

lint:
	@pre-commit run --all-files

clean:
	rm -rf logs cache .pytest_cache .mypy_cache
	clear
