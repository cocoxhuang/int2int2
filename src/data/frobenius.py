"""Frobenius traces a_p(E) computed from a global minimal Weierstrass model.

The dataset used in Appendix B is built from LMFDB's ``ainvs`` column alone;
the a_p values are not stored in LMFDB and are recomputed here.
"""
import numpy as np

PRIMES_UNDER_100 = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97)


def _root_count_table(p):
    """table[d] = #{y in F_p : y^2 + by + c = 0} for a quadratic of discriminant d."""
    squares = {(y * y) % p for y in range(p)}
    return np.array([1 if d == 0 else (2 if d in squares else 0) for d in range(p)],
                    dtype=np.int64)


def ap_matrix(ainvs, primes=PRIMES_UNDER_100):
    """Frobenius traces for many curves at once.

    ainvs : integer array of shape (n, 5), global minimal models [a1, a2, a3, a4, a6].
    Returns an int64 array of shape (n, len(primes)).

    For every p,  a_p = p - A,  where A is the number of affine points of the
    reduced model over F_p.  The single formula covers both cases:

        good p :  #E(F_p)    = A + 1   =>  a_p = p + 1 - #E(F_p) = p - A
        bad  p :  #E_ns(F_p) = A       =>  a_p = p - #E_ns(F_p)  = p - A

    (at a bad prime the unique singular point drops out and infinity comes in),
    so additive / split / non-split reduction correctly yield a_p = 0 / +1 / -1.
    """
    ainvs = np.asarray(ainvs, dtype=np.int64)
    if ainvs.ndim != 2 or ainvs.shape[1] != 5:
        raise ValueError(f"expected shape (n, 5), got {ainvs.shape}")
    n = ainvs.shape[0]
    out = np.empty((n, len(primes)), dtype=np.int64)

    for j, p in enumerate(primes):
        a1, a2, a3, a4, a6 = (ainvs[:, k] % p for k in range(5))
        A = np.zeros(n, dtype=np.int64)
        if p == 2:
            # the discriminant shortcut degenerates in characteristic 2
            for x in range(2):
                for y in range(2):
                    lhs = y * y + a1 * x * y + a3 * y
                    rhs = x ** 3 + a2 * x * x + a4 * x + a6
                    A += ((lhs - rhs) % p == 0)
        else:
            tab = _root_count_table(p)
            for x in range(p):
                b = (a1 * x + a3) % p
                c = (x ** 3 + a2 * x * x + a4 * x + a6) % p
                # y^2 + b y - c = 0  has discriminant  b^2 + 4c
                A += tab[(b * b + 4 * c) % p]
        out[:, j] = p - A
    return out
