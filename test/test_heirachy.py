class Parent():
    def __init__(self,name,*args):
        self.name = name
        print("i am parent,name=", name)

class Child(Parent):
    def __init__(self, name, *args):
        super().__init__(name,args)
        print("i am child,name=",name)


class GrandChild(Child):
    def __init__(self, name, *args):
        super().__init__(name,args)
        print("i am grandchild,name=", name)


gc = GrandChild("grand_child_hello","...")
print("finally, got:",gc.name)

