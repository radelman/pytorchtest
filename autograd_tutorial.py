

import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights


def main1() -> None:
    print("hello, world!")





    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    data = torch.rand(1, 3, 64, 64)
    labels = torch.rand(1, 1000)



    #print(model)
    #print(data)
    #print(labels)

    print(data.shape)
    print(labels.shape)

    prediction = model(data) # forward pass

    print(prediction.shape)

    loss = (prediction - labels).sum()
    loss.backward() # backward pass


    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    optim.step() #gradient descent


def main2() -> None:

    a = torch.tensor([[2., 3.], [1., 2.]], requires_grad=True)
    b = torch.tensor([[6., 4.], [9., 7.]], requires_grad=True)

    Q = 3*a**3 - b**2

    print(Q)


    # Q.backward(torch.ones_like(Q))
    Q.backward(torch.ones_like(Q), create_graph=True)


    print(a.grad)
    print(b.grad)


    a.grad = None
    b.grad = None


    # have to add create_graph=True above to be able to do this again, or
    # uncomment this line to recalculate Q from scratch.  why?
    # Q = 3*a**3 - b**2


    Q.sum().backward()

    print(a.grad)
    print(b.grad)


def main3() -> None:

    a = torch.tensor([[2., 3.], [1., 2.]], requires_grad=True)
    b = torch.tensor([[6., 4.], [9., 7.]], requires_grad=True)

    print(a)
    print(b)

    Q = 3*a**3 - b**2

    print(Q)


    R = Q.sum()

    R.backward()


    print(a.grad)
    print(b.grad)
    print(Q.grad)


    a.grad = None
    b.grad = None
    Q.grad = None


    print("changing a")

    a = torch.tensor([[1., 1.], [1., 1.]], requires_grad=True)

    print(a)
    print(b)

    Q = 3*a**3 - b**2

    print(Q)


    R = Q.sum()

    R.backward()


    print(a.grad)
    print(b.grad)
    print(Q.grad)


def main4() -> None:

    # Optional Reading - Vector Calculus using autograd


    # see https://math.stackexchange.com/questions/222894/how-to-take-the-gradient-of-the-quadratic-form

    Anp = np.array([[5.0, 2.0], [1.0, 3.0]], dtype=np.float64)
    xnp = np.array([[1.0], [5.0]], dtype=np.float64)

    print(Anp)
    print(xnp)


    print(xnp.T @ (Anp @ xnp))

    print((Anp + Anp.T) @ xnp)


    h = 0.00001

    y0 = xnp.T @ (Anp @ xnp)
    xnp[0] += h
    y1 = xnp.T @ (Anp @ xnp)
    xnp[0] -= h
    print((y1 - y0) / h)

    y0 = xnp.T @ (Anp @ xnp)
    xnp[1] += h
    y1 = xnp.T @ (Anp @ xnp)
    xnp[1] -= h
    print((y1 - y0) / h)


    A = torch.tensor(Anp, requires_grad=True)
    x = torch.tensor(xnp, requires_grad=True)






    y = x.T @ (A @ x)

    print(y)


    y.backward()

    print(A.grad)
    print(x.grad)


    print((A + A.T) @ x)



def main5() -> None:

    # Optional Reading - Vector Calculus using autograd, cont'd


    Anp = np.array([[5.0, 2.0], [1.0, 3.0]], dtype=np.float64)
    xnp = np.array([[1.0], [5.0]], dtype=np.float64)

    print(Anp)
    print(xnp)


    A = torch.tensor(Anp, requires_grad=True)
    x = torch.tensor(xnp, requires_grad=True)


    y = A @ x


    print(y)


    # torch does not do backward propagation on a vector-valued function, y.
    # it requires you to either explictly aggregate the function as a scalar,
    # v(y) (via, e.g., sum), or do it implicitly by passing in dvdy


    dvdznp = np.array([[1.0], [1.0]], dtype=np.float64)

    dvdz = torch.tensor(dvdznp)

    y.backward(dvdz)

    print(A.grad)
    print(x.grad)






if __name__ == "__main__":
    # main1()
    # main2()
    # main3()
    # main4()
    main5()
