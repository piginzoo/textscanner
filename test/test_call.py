
def call1(p1):
    print("call1!")
    print(p1)

def call2(p1,p2):
    print("call2!")
    print(p1)
    print(p2)

def call3(p_list):
    print("call3!")
    print(p_list)


def test_func(c,*param):
    print(type(param))
    c(*param)

test_func(call1, "aaaa")
test_func(call2, "bbbb","cccc")
test_func(call3, ["bbbb","cccc"])