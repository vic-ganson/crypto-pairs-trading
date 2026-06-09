# crypto-pairs-trading
A statistical arbitrage strategy for cryptocurrencies using cointegration-based pairs selection and a Kalman filter for dynamic hedge ratio estimation.

The core idea is pairs trading: we identify two cryptocurrencies whose prices move together over the long run and trade them when they drift apart (expecting that they'll revert to their historical relationship).
