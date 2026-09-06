# Controlled data contract

The current paper code is intentionally tied to the controlled RestaurantFastFood/Buffalo Phase-D products.
Core method classes receive thermal state/forcing arrays; ScaleBridge campaign/run discovery is isolated in the data adapters.

`Q_AC` is signed delivered HVAC thermal heat [W]. `P_HVAC` is electrical HVAC power [W] and never enters the thermal ODE.
