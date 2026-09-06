# Method progression

- Inverse PINN-RC: identify physical RC parameters from a physics-informed trajectory surrogate; deploy the physical RC ODE.
- Neural ODE: data-only continuous-time neural vector field.
- Base PINODE: integrate the raw neural derivative and penalize full RC residuals softly at actual RK4 RHS stages.
- EBP-PINODE: project the raw derivative onto exact zone total-energy constraints at every RK4 RHS stage, then integrate the projected derivative; retain only independent 2C internal physics as soft residuals.
