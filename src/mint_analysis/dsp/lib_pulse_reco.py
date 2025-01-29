"""
This module contains helper functions for pulse analysis.
"""

from __future__ import annotations
import numpy as np


class pulse_analysis:

    def find_new_t0(self, baseline:float, zres:np.array, x:np.array) -> dict:
        integrals = {}
        shifted_res = zres - baseline
        multiply = shifted_res * np.roll(shifted_res, 1)
        sign = np.sign(multiply)
        sign[0] = 1
        peak = False
        current_integral = 0
        current_integral_start = 0

        for i, s in enumerate(sign):
            if s < 0:  # When the sign changes (indicating a peak)
                if not peak:
                    current_integral = 0
                    current_integral_start = x[i]
                elif peak:
                    integrals[(current_integral_start, x[i])] = current_integral
                peak = not peak

            if peak:  # While inside a peak, accumulate the integral
                current_integral += zres[i] * (x[1] - x[0])  # Assuming uniform spacing

        return integrals

    def error_on_fit(self, reco_t0:np.array, loc:np.array) -> list:
        # quantify the error on the fit
        error_on_fit = []
        error = 0
        for a, b in enumerate(reco_t0):
            error = 0 if (loc[a] - b) <= 0 else (loc[a] - b) / loc[a] * 100
            error_on_fit.append(error)
        return error_on_fit
