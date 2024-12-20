## Visualizing Scheme Theory with Python
Slide 1: Introduction to Schemes

Schemes are fundamental objects in algebraic geometry, generalizing the concept of algebraic varieties. They provide a more flexible framework for studying geometric objects algebraically.

```python
import sympy as sp

# Define symbolic variables
x, y = sp.symbols('x y')

# Define an affine scheme (algebraic variety)
scheme = sp.Eq(x**2 + y**2 - 1, 0)

print(f"Equation of the circle (affine scheme): {scheme}")
```

Slide 2: Spec of a Ring

The Spec (spectrum) of a ring is the set of all prime ideals of the ring, forming the building blocks of schemes.

```python
from sympy import ZZ

def prime_ideals(n):
    return [p for p in range(2, n+1) if ZZ.is_prime(p)]

ring = range(20)
spec = prime_ideals(20)

print(f"Spec of Z/20Z: {spec}")
```

Slide 3: Sheaves of Rings

Sheaves are a way to associate data (like functions) to the open sets of a topological space, satisfying certain compatibility conditions.

```python
class Sheaf:
    def __init__(self, base_ring):
        self.base_ring = base_ring
        self.sections = {}
    
    def define_section(self, open_set, function):
        self.sections[open_set] = function
    
    def restrict(self, from_set, to_set):
        if to_set.issubset(from_set):
            return self.sections[from_set]
        raise ValueError("Invalid restriction")

# Example usage
R = Sheaf("base_ring")
R.define_section("U", lambda x: x**2)
R.define_section("V", lambda x: x+1)

print(R.sections)
```

Slide 4: Locally Ringed Spaces

A locally ringed space is a topological space equipped with a sheaf of rings, where the stalks are local rings.

```python
class LocallyRingedSpace:
    def __init__(self, topological_space, structure_sheaf):
        self.space = topological_space
        self.sheaf = structure_sheaf
    
    def stalk_at_point(self, point):
        # In practice, this would compute the stalk
        return f"Stalk at {point}"

# Example
X = LocallyRingedSpace("Topological Space", Sheaf("OX"))
print(X.stalk_at_point("p"))
```

Slide 5: Affine Schemes

Affine schemes are the building blocks of general schemes, constructed from the Spec of a ring.

```python
def affine_scheme(ring):
    spec = set()
    for element in ring:
        if is_prime_ideal(element):
            spec.add(element)
    return spec

def is_prime_ideal(ideal):
    # Simplified prime ideal check
    return ideal != 0 and ideal != 1

ring = range(10)
A = affine_scheme(ring)
print(f"Affine scheme of ring: {A}")
```

Slide 6: Gluing Schemes

Schemes can be constructed by gluing together affine schemes, similar to how manifolds are built from coordinate charts.

```python
class Scheme:
    def __init__(self):
        self.affine_pieces = {}
    
    def add_affine_piece(self, name, affine_scheme):
        self.affine_pieces[name] = affine_scheme
    
    def glue(self, piece1, piece2, gluing_map):
        # Implement gluing logic here
        pass

# Example usage
X = Scheme()
X.add_affine_piece("U", affine_scheme(range(5)))
X.add_affine_piece("V", affine_scheme(range(5, 10)))
X.glue("U", "V", lambda x: x+5)

print(X.affine_pieces)
```

Slide 7: Morphisms of Schemes

Morphisms between schemes are structure-preserving maps that respect the locally ringed space structure.

```python
def scheme_morphism(source, target, f):
    def pullback(s):
        return lambda x: s(f(x))
    
    new_sheaf = {}
    for open_set, section in target.structure_sheaf.items():
        new_sheaf[f.inverse_image(open_set)] = pullback(section)
    
    return new_sheaf

# Example (simplified)
source = {"structure_sheaf": {"U": lambda x: x**2}}
target = {"structure_sheaf": {"V": lambda x: x+1}}
f = lambda x: 2*x

morphism = scheme_morphism(source, target, f)
print(morphism)
```

Slide 8: Projective Schemes

Projective schemes are important in algebraic geometry, generalizing projective varieties and allowing for a more natural treatment of points at infinity.

```python
import sympy as sp

def projective_scheme(homogeneous_poly):
    x, y, z = sp.symbols('x y z')
    return sp.homogeneous_order(homogeneous_poly)

# Example: projective conic
conic = x**2 + y**2 - z**2

print(f"Degree of projective conic: {projective_scheme(conic)}")
```

Slide 9: Fiber Products of Schemes

Fiber products are a fundamental construction in scheme theory, generalizing the notion of pullback from category theory.

```python
def fiber_product(X, Y, S, f, g):
    def pullback(x, y):
        return f(x) == g(y)
    
    XxY = [(x, y) for x in X for y in Y]
    return [p for p in XxY if pullback(p[0], p[1])]

# Example
X = [1, 2, 3]
Y = [2, 3, 4]
S = [0, 1, 2]
f = lambda x: x % 2
g = lambda y: y % 2

XxS_Y = fiber_product(X, Y, S, f, g)
print(f"Fiber product: {XxS_Y}")
```

Slide 10: Cohomology of Schemes

Cohomology is a powerful tool in algebraic geometry, providing invariants for schemes and a framework for proving deep results.

```python
def cech_cohomology(scheme, cover):
    def boundary(cochain):
        # Implement boundary operator
        pass
    
    def cocycles(degree):
        # Compute cocycles
        pass
    
    def coboundaries(degree):
        # Compute coboundaries
        pass
    
    return lambda i: cocycles(i) / coboundaries(i)

# Example usage (simplified)
scheme = {"U": set(), "V": set()}
cover = ["U", "V"]
H = cech_cohomology(scheme, cover)

print(f"H^1: {H(1)}")
```

Slide 11: Derived Categories of Schemes

Derived categories provide a framework for homological algebra on schemes, allowing for a more flexible treatment of complexes of sheaves.

```python
class Complex:
    def __init__(self, objects, differentials):
        self.objects = objects
        self.differentials = differentials

def derived_category(scheme):
    def quasi_isomorphism(f):
        # Check if f induces isomorphism on cohomology
        pass
    
    def localize(complexes, quasi_isomorphisms):
        # Implement localization
        pass
    
    return localize

# Example usage (simplified)
D = derived_category("scheme")
complex = Complex(["O_X", "O_X(1)", "O_X(2)"], [lambda x: x, lambda x: x**2])
print(f"Object in derived category: {complex.objects}")
```

Slide 12: Moduli Spaces as Schemes

Moduli spaces parameterize geometric objects and can often be given the structure of a scheme, providing a rich interplay between geometry and algebra.

```python
def hilbert_scheme(n, P):
    def hilbert_polynomial(F):
        # Compute Hilbert polynomial
        pass
    
    def flat_family(X):
        # Check if X is a flat family
        pass
    
    return lambda X: hilbert_polynomial(X) == P and flat_family(X)

# Example usage (simplified)
H = hilbert_scheme(3, lambda t: t**2 + 1)
X = "some_scheme"
print(f"Is X in the Hilbert scheme? {H(X)}")
```

Slide 13: Étale Cohomology and l-adic Sheaves

Étale cohomology provides a cohomology theory for schemes in positive characteristic, analogous to singular cohomology in topology.

```python
def etale_cohomology(X, l):
    def etale_cover(U):
        # Compute étale cover
        pass
    
    def l_adic_sheaf(F):
        # Construct l-adic sheaf
        pass
    
    def compute_cohomology(cover, sheaf):
        # Compute cohomology
        pass
    
    return compute_cohomology(etale_cover(X), l_adic_sheaf("F"))

# Example usage (simplified)
X = "smooth_projective_variety"
l = 2  # Prime different from characteristic
H_et = etale_cohomology(X, l)
print(f"Étale cohomology groups: {H_et}")
```

Slide 14: Additional Resources

1. Hartshorne, R. (1977). Algebraic Geometry. Springer-Verlag. ArXiv: [https://arxiv.org/abs/alg-geom/9711008](https://arxiv.org/abs/alg-geom/9711008) (related survey)
2. Grothendieck, A. & Dieudonné, J. (1960-1967). Éléments de géométrie algébrique. Publications Mathématiques de l'IHÉS. ArXiv: [https://arxiv.org/abs/math/0206203](https://arxiv.org/abs/math/0206203) (modern interpretation)
3. Vakil, R. The Rising Sea: Foundations of Algebraic Geometry. Available at: [http://math.stanford.edu/~vakil/216blog/FOAGnov1817public.pdf](http://math.stanford.edu/~vakil/216blog/FOAGnov1817public.pdf)
4. Stacks Project Authors. Stacks Project. [https://stacks.math.columbia.edu/](https://stacks.math.columbia.edu/)

These resources provide in-depth treatments of scheme theory and related topics in algebraic geometry.

