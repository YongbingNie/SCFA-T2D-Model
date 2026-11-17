# SCFA-T2D-Model

This repository contains the code used for the mathematical analysis and numerical simulations of a within-host model of type 2 diabetes (T2D) incorporating short-chain fatty acid (SCFA) regulation.

## Code files

- `SCFA_structural_identifiability_G.jl`: Structural identifiability analysis under glucose-only observation.
- `SCFA_parameter_fitting.py`: Subject-specific parameter estimation using longitudinal glucose data from 11 subjects.
- `SCFA_PRCC.py`: Global sensitivity analysis using partial rank correlation coefficients.
- `SCFA_bifurcation.py`: Numerical bifurcation analysis with respect to SCFA supply.
- `SCFA_intervention.py`: Simulations of SCFA supply and dietary glucose interventions.

## Data

The longitudinal glucose data used for parameter estimation are provided in the associated manuscript and included in the corresponding code.

## Requirements

Python dependencies are listed in `requirements.txt`. Structural identifiability analysis requires Julia and the StructuralIdentifiability.jl package.

## Usage

Run the corresponding script to reproduce each analysis. The scripts are organized according to the numerical methods and results presented in the manuscript.
