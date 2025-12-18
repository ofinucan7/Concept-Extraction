Concept Extraction from University Lecture Slides with LLMs

This repository contains the code, prompts, and analysis pipelines for a research project investigating how reliably large language models (LLMs) can extract explicit instructional concepts from university lecture slide decks.

The work evaluates LLM-based concept extraction against expert human annotations and analyzes which slide-level linguistic properties most strongly influence extraction performance.

Project Overview

Lecture slides are one of the primary ways instructors communicate core concepts to students, yet manually extracting those concepts is time-consuming, subjective, and difficult to scale.

This project asks two core questions:

Can LLMs reliably extract instructional concepts from lecture slides?

What properties of slides make concept extraction easier or harder for LLMs?

To answer these, we design and evaluate two minimalist LLM pipelines and compare their outputs against gold-standard human annotations created under a shared, rule-based codebook.

Models
Model 1 — Text-Only Pipeline

Operates solely on text extracted from slide decks

Directly prompts an LLM to extract explicitly defined concepts

Uses no visual information

Produces moderate and consistent performance across courses

Model 2 — Multimodal Pipeline

Combines slide text with LLM-generated summaries derived from slide screenshots

Uses a two-stage process (image → summary → concept extraction)

Performs substantially worse than the text-only model

Suggests that naive inclusion of visuals introduces noise rather than clarity

Key Findings

Text-only extraction outperforms multimodal extraction in both accuracy and reliability

Slides with higher linguistic structure yield better concept extraction

Three slide-level predictors explain most performance variation:

POS entropy (positive effect)

Verbosity penalty (negative effect)

Slide compression potential (marginal effect)

Overall, the results suggest that clarity, conciseness, and structured language matter more than modality richness for grounded concept extraction.

Concept Annotation Codebook

All annotations — human and model-generated — follow a shared concept annotation codebook designed to make concept extraction reproducible and rule-based.

Core principles include:

Strict grounding in slide text (no inference or hallucination)

Explicit definitions, emphasis, or example usage

Controlled granularity (no oversplitting or vague terms)

Canonical forms only (no near-synonyms or duplicates)

Exclusion of generic, structural, or implementation-only terms

This framing treats concept extraction as an evidence-based annotation task, not summarization.
