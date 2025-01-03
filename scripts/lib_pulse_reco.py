import math
import numpy as np
import matplotlib.pyplot as plt
import numba as nb


class calculate_reco_waveform:

    def res(self, x, y):
    
        def nnls(A, b, maxiter=None, tol=None):

            #@nb.jit(nopython=True, cache=False)  # cache=False only for performance comparison
            def numba_ix(arr, rows, cols):
                """
                Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays.
                from https://github.com/numba/numba/issues/5894#issuecomment-974701551
                :param arr: 2D array to be indexed
                :param rows: Row indices
                :param cols: Column indices
                :return: 2D array with the given rows and columns of the input array
                """
                one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
                for i, r in enumerate(rows):
                    start = i * len(cols)
                    one_d_index[start: start + len(cols)] = cols + arr.shape[1] * r

                arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
                slice_1d = np.take(arr_1d, one_d_index)
                return slice_1d.reshape((len(rows), len(cols)))
        
            """
            based on scipy implementation. Which in turn is based on
            the algorithm given in  :doi:`10.1002/cem.889`
            """
            m, n = A.shape

            AtA = np.transpose(A) @ A
            Atb = b @ A  # Result is 1D - let NumPy figure it out

            if not maxiter:
                maxiter = 3*n
            if tol is None:
                tol = 10 * max(m, n) * np.spacing(1.)

            # Initialize vars
            x = np.zeros(n, dtype=np.float64)
            s = np.zeros(n, dtype=np.float64)
            # Inactive constraint switches
            P = np.zeros(n, dtype=bool)
            Pidx = np.arange(0,len(P),1,dtype=int)

            # Projected residual
            w = Atb.copy().astype(np.float64)  # x=0. Skip (-AtA @ x) term

            # Overall iteration counter
            # Outer loop is not counted, inner iter is counted across outer spins
            iter = 0

            while (not P.all()) and (w[~P] > tol).any():  # B
                # Get the "most" active coeff index and move to inactive set
                k = np.argmax(w * (~P))  # B.2
                P[k] = True  # B.3

                # Iteration solution
                s[:] = 0.
                # B.4
                s[P] = np.linalg.solve(numba_ix(AtA,Pidx[P],Pidx[P]), Atb[P])

                # Inner loop
                while (iter < maxiter) and (s[P].min() <= 0):  # C.1
                    iter += 1
                    inds = P * (s <= 0)
                    alpha = (x[inds] / (x[inds] - s[inds])).min()  # C.2
                    x *= (1 - alpha)
                    x += alpha*s
                    P[x <= tol] = False
                    s[P] = np.linalg.solve(numba_ix(AtA,Pidx[P],Pidx[P]), Atb[P])
                    s[~P] = 0  # C.6

                x[:] = s[:]
                w[:] = Atb - AtA @ x

                if iter == maxiter:
                    return x

            return x
        # Gumbel distribution (https://en.wikipedia.org/wiki/Gumbel_distribution)
        def gumbel_pdf(x,mu,sigma):
            beta = sigma*(np.sqrt(6)/np.pi)
            z=(x-mu)/beta
            return (1/beta)*np.exp(-(z+np.exp(-1*z)))
            
        A = np.zeros((len(x),len(x)))
        for i in range(len(x)):
            A[:,i] = (gumbel_pdf(x,x[i],2))
        res = nnls(A,y,1000)
        zres = np.zeros(len(x))
        for i in range(len(zres)):
            zres += res[i]*gumbel_pdf(x,x[i],2)
        return zres

class pulse_analysis:

    def find_new_t0(self, baseline, zres, x):
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
    
    def error_on_fit(self, reco_t0, loc):
    # quantify the error on the fit
        error_on_fit = []
        error = 0
        for a, b in enumerate(reco_t0):
            if (loc[a] - b) <= 0:
                error = 0
            else:
                error = (loc[a] - b)/loc[a]*100
            error_on_fit.append(error)
        return error_on_fit
