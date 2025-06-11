# Makefile for Concept Extraction Project

.PHONY: help install test clean run-all-subjects demo

help:
	@echo "Available commands:"
	@echo "  install         - Install required packages"
	@echo "  test           - Run tests on all subjects"
	@echo "  demo           - Run demo with ancient history"
	@echo "  run-all-subjects - Run extraction on all available subjects"
	@echo "  clean          - Clean up generated files"
	@echo "  notebook       - Start Jupyter notebook"

install:
	pip install -r requirements.txt

demo:
	python main.py --subject ancient_history
	@echo "\nDemo complete! Check output_concepts.csv for results."

test:
	python main.py --subject ancient_history
	python main.py --subject economics  
	python main.py --subject mathematics
	python main.py --subject physics
	@echo "\nAll subjects tested successfully!"

run-all-subjects:
	@echo "Running concept extraction on all subjects..."
	python main.py --subject ancient_history
	@mv output_concepts.csv output_ancient_history.csv
	python main.py --subject economics
	@mv output_concepts.csv output_economics.csv  
	python main.py --subject mathematics
	@mv output_concepts.csv output_mathematics.csv
	python main.py --subject physics
	@mv output_concepts.csv output_physics.csv
	@echo "All extractions complete! Check output_*.csv files."

clean:
	rm -f output_*.csv
	rm -rf __pycache__/
	rm -f *.pyc

notebook:
	jupyter notebook solution.ipynb
