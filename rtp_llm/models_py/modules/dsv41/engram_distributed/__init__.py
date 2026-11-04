# SPDX-License-Identifier: Apache-2.0
"""Distributed Engram storage for an immutable startup cohort.

startup: derive the cohort and exchange capabilities before worker launch.
bootstrap: validate the staged package, attach mappings and verify GPU samples.
model_attach: publish all model-local layers and retain storage/Graph ownership.
storage/mapping: own imported views, reader leases, synchronization and cleanup.

This package does not provide online resharding or per-host live replacement.
Importing it performs no discovery, allocation, native build or CUDA work.
"""
