from pprint import pprint
import numpy as np
import sympy
import random

def compute_transition_invariants(net):
  """
  We compute the NUllspace of the incidence matrix and obtain the place-invariants.
  :param net: Petri Net of which we want to know the place invariants.
  :return: Set of place invariants of the given Petri Net.
  """

  def compute_incidence_matrix(net):
    # TODO change to t_components
    """
    Given a Petri Net, the incidence matrix is computed. An incidence matrix has n rows (places) and m columns
    (transitions).
    :param net: Petri Net object
    :return: Incidence matrix
    """
    n = len(net.transitions)
    m = len(net.places)
    C = np.zeros((n, m))
    i = 0
    transition_list = sorted(list(net.transitions), key=lambda transition: str(transition))
    place_list = sorted(list(net.places), key=lambda place: str(place))
    while i < m:
        p = place_list[i]
        for in_arc in p.in_arcs:
            # arcs that go to place
            C[transition_list.index(in_arc.source), i] -= 1
        for out_arc in p.out_arcs:
            # arcs that lead away from place
            C[transition_list.index(out_arc.target), i] += 1
        i += 1
    np.set_printoptions(edgeitems=30, linewidth=100000,
                        formatter=dict(float=lambda x: f"{int(x):>2}"))
    return C

  def extract_basis_vectors(incidence_matrix):
    """
    The name of the method describes what we want t achieve. We calculate the nullspace of the transposed identity matrix.
    :param incidence_matrix: Numpy Array
    :return: a collection of numpy arrays that form a base of transposed A
    """
    # To have the same dimension as described as in https://www7.in.tum.de/~esparza/fcbook-middle.pdf and to get the correct nullspace, we have to transpose
    A = np.transpose(incidence_matrix)
    # exp from book https://www7.in.tum.de/~esparza/fcbook-middle.pdf
    x = sympy.Matrix(A).nullspace()
    # TODO: Question here: Will x be always rational? Depends on sympy implementation. Normaly, yes, we we will have rational results
    x = np.array(x).astype(np.float64)
    return x

  A = compute_incidence_matrix(net)
  return extract_basis_vectors(A)
