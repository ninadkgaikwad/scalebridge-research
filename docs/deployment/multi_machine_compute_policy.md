# ScaleBridge Multi-Machine Development and Compute Policy

Date: 2026-05-31  
Project: ScaleBridge Research  
Repository: scalebridge-research  

## Purpose

This document defines how ScaleBridge code, environments, and results should be managed across:

- dev-laptop,
- home-pc,
- lab-pc,
- WSU Kamiak HPC.

The goal is to use the laptop for development and the other machines for compute while keeping code and final outputs synchronized.

## Machine Roles

| Machine | OS | Access | Role |
|---|---|---|---|
| dev-laptop | Windows | local | development, GitHub Desktop, orchestration |
| home-pc | Windows | Chrome Remote Desktop from dev-laptop | compute |
| lab-pc | Windows | Chrome Remote Desktop from dev-laptop | compute |
| Kamiak HPC | Linux / SLURM | MobaXterm from dev-laptop | batch compute |

## Code Synchronization

Windows machines may use Dropbox to sync the repository folder.

GitHub remains the version-control source of truth.

Recommended workflow:

```text
edit on dev-laptop
→ commit with Git
→ push to GitHub
→ allow Dropbox to sync to Windows compute machines
→ run compute jobs
→ sync outputs back through Dropbox