#################################
# Your name:
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):

    def sample_from_D(self, m):
        xs = np.sort(np.random.uniform(0, 1, m))
        ys = np.where(
            (xs <= 0.2) | ((xs >= 0.4) & (xs <= 0.6)) | (xs >= 0.8),
            np.random.binomial(1, 0.8, m),
            np.random.binomial(1, 0.1, m)
        )
        return np.column_stack((xs, ys))

    def true_error(self, ivals):
        good = [(0.0, 0.2), (0.4, 0.6), (0.8, 1.0)]
        bad  = [(0.2, 0.4), (0.6, 0.8)]
        ig = sum(self._overlap(ivals, r) for r in good)
        ib = sum(self._overlap(ivals, r) for r in bad)
        return -0.6 * ig + 0.8 * ib + 0.52

    def _overlap(self, ivals, region):
        a, b = region
        return sum(max(0.0, min(b, u) - max(a, l)) for l, u in ivals)

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        ms = list(range(m_first, m_last + 1, step))
        results = np.zeros((len(ms), 2))
        for idx, m in enumerate(ms):
            emp_runs, true_runs = [], []
            for _ in range(T):
                data = self.sample_from_D(m)
                xs, ys = data[:, 0], data[:, 1]
                best_ivals, emp_count = intervals.find_best_interval(xs, ys, k)
                emp_runs.append(emp_count / m)
                true_runs.append(self.true_error(best_ivals))
            results[idx] = [np.mean(emp_runs), np.mean(true_runs)]
        plt.figure()
        plt.plot(ms, results[:, 0], label='Empirical')
        plt.plot(ms, results[:, 1], label='True')
        plt.xlabel('m')
        plt.ylabel('Error')
        plt.title(f'Errors vs m  (k={k}, T={T})')
        plt.legend()
        plt.savefig('b_errors_vs_m.png')
        plt.show()
        return results

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        data = self.sample_from_D(m)
        xs, ys = data[:, 0], data[:, 1]
        ks = list(range(k_first, k_last + 1, step))
        emp_errs, true_errs = [], []
        for k in ks:
            best_ivals, emp_count = intervals.find_best_interval(xs, ys, k)
            emp_errs.append(emp_count / m)
            true_errs.append(self.true_error(best_ivals))
        plt.figure()
        plt.plot(ks, emp_errs, label='Empirical')
        plt.plot(ks, true_errs, label='True')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.title('Errors vs k')
        plt.legend()
        plt.savefig('c_errors_vs_k.png')
        plt.show()
        return ks[int(np.argmin(emp_errs))]

    def cross_validation(self, m):
        data = self.sample_from_D(m)
        np.random.shuffle(data)
        split = int(0.8 * m)
        train = data[:split][data[:split, 0].argsort()]
        val   = data[split:]
        xs_tr, ys_tr = train[:, 0], train[:, 1]
        xs_val, ys_val = val[:, 0], val[:, 1]
        val_errs = []
        for k in range(1, 11):
            best_ivals, _ = intervals.find_best_interval(xs_tr, ys_tr, k)
            preds = np.array([
                1 if any(l <= x <= u for l, u in best_ivals) else 0
                for x in xs_val
            ])
            val_errs.append(np.mean(preds != ys_val))
        best_k = int(np.argmin(val_errs)) + 1
        return best_k

    #################################
    # Place for additional methods
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.cross_validation(1500)
