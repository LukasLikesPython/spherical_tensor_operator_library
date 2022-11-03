# stolpy - a spherical tensor operator library for python
A python package that can be used to simplify equations using 
spherical tensor operators by automatically applying relations 
from reference [A. Varshalovich, A. N. Moskalev, and V. K. Khersonskii, Quantum Theory of Angular
Momentum. Singapore: World Scientific, 1988].

This project is part of my master thesis at the UoO. I will add the notes after handing in my thesis.


## Authors

- [Lukas Huth](https://github.com/LukasLikesPython)

## Installation
Install my-project with pip
```bash
  pip install stolpy
```

## Requirements

* python>=3.9
* sympy>=1.7


## Usage Example
The example is for two-body contact operators from chiral EFT (nuclear physics).

First define the spaces for the operators and states
```python
from stolpy.tensor_space import TensorSpace

rel_space = TensorSpace('rel', 0)
spin_space = TensorSpace('spin', 1)
```
Define all spaces you need and order them in the way you want the operators ordered.
Do not use the same name for a space twice.

Next we initialize our operators. We use their vector representation and the code will transform them into tensor operators.
For this example, we use the tensor operator in q (C6) at NLO (next to leading order).
```python
from stolpy.tensor_transformation import TensorFromVectors
from stolpy.tensor_operator import TensorOperator
from sympy import Symbol

q = TensorOperator(rank=1, symbol=Symbol("q"), space=rel_space)
sig1 = TensorOperator(rank=1, symbol=Symbol("sig1"), space=spin_space)
sig2 = TensorOperator(rank=1, symbol=Symbol("sig2"), space=spin_space)
tensor_op = TensorFromVectors.scalar_product(q, sig1).\
                couple(TensorFromVectors.scalar_product(q, sig2), 0, 1)
```
As you can see, we need to provide a rank (here rank=1 since all operators are vectors), a symbol (use Sympy.Symbol, however a string will also work), and the space (see above).

We can print the operator
```python
print(tensor_op)
```
```
3 * {{q_1 x sig1_1}_0 x {q_1 x sig2_1}_0}_0
```

If we are only interested in the tensor operator itself, we can use the tensor_algebra package to simplify the expression
```python
# optional
from stolpy.tensor_algebra import TensorAlgebra

decoupled_op = TensorAlgebra.recouple(tensor_op)
print(decoupled_op)
```
```
1 * {{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0 + sqrt(5) * {{q_1 x q_1}_2 x {sig1_1 x sig2_1}_2}_0
```
As one can see, the decoupled operator's sub-tensors are reordered according to the space in which they act.
Note that this step is optional, the functionality we discuss next automatically performs this step.

Another possibility is to create the matrix elements for this operator. For this we have to define the states first
```python
from stolpy.quantum_states import BasicState

s = BasicState(Symbol("s"), spin_space)
l = BasicState(Symbol("l"), rel_space, Symbol("p"))
ket = l.couple(s, Symbol("j"))
sp = BasicState(Symbol("s'"), spin_space)
lp = BasicState(Symbol("l'"), rel_space, Symbol("p'"))
bra = lp.couple(sp, Symbol("j'"))
```
Here, s = spin, l = orbital angular momentum, and j = total angular momentum.

We can now define the matrix element
```python
from stolpy.tensor_evaluate import MatrixElement

me = MatrixElement(bra, ket, tensor_op)
print(me)
```
```
<p'j'(l's')m_j'|{{q_1 x q_1}_0 x {sig1_1 x sig2_1}_0}_0|pj(ls)m_j> + sqrt(5) * <p'j'(l's')m_j'|{{q_1 x q_1}_2 x {sig1_1 x sig2_1}_2}_0|pj(ls)m_j>
```
As one can see, the simplification is done automatically during the construction of the matrix element. We can now
break down the states and operators by using the full_decouple method of the MatrixElement class.
```python
decoupled_me = me.full_decouple()
print(decoupled_me)
```
```
(-1)**(j' + l + s')*KroneckerDelta(j, j') * SixJ(l' s' j'; s l 0) * <p'l'||{q_1 x q_1}_0||pl><s'||{sig1_1 x sig2_1}_0||s> + (-1)**(j' + l + s' + 2)*KroneckerDelta(j, j') * SixJ(l' s' j'; s l 2) * <p'l'||{q_1 x q_1}_2||pl><s'||{sig1_1 x sig2_1}_2||s>
```
This breaks down the equation to factors and reduced matrix elements. Apply the Wigner-Eckart Theorem (WET) to calculate the reduced matrix elements manually. One has to do this once and can use the result independent of any projections. 

While this already provides a general expression, one can go one step further and evaluate the expression (up to reduced matrix elements). In order to do so, we need to define a dictionary that contains the symbols in the equation and the values for which you want to substitute them.
```python
subsdict = {
    Symbol("l"): 1,
    Symbol("l'"): 1,
    Symbol("s"): 1,
    Symbol("s'"): 1,
    Symbol("j"): 2,
    Symbol("j'"): 2,
}
result = me.evaluate(subsdict)
print(result)
```
```
<1||{sig1_1 x sig2_1}_0||1>*<p'1||{q_1 x q_1}_0||p1>/3 + <1||{sig1_1 x sig2_1}_2||1>*<p'1||{q_1 x q_1}_2||p1>/30
```
The only thing left to do is to evaluate the reduced matrix elements using the WET. That's it. 

## Message from the author
I hope this helps your research. Let me know about any bugs on https://github.com/LukasLikesPython/spherical_tensor_operator_library/issues. 
I am also happy about any kind of feedback. Feel free to add to this if you need additional functions. 

Cheers

Lukas

## Legal

This work is published under the  GNU GENERAL PUBLIC LICENSE Version 3. See the LICENSE file or https://www.gnu.org/licenses/ for details.

### Warranty
  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

### Liability
  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.