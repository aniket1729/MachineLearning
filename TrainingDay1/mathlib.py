def add(a,b):
    print("Added = ", a+b)
    
def minus(a,b):
    print("Sub = ", a-b)
    
def divide(a,b):
    print("Div = ", a/b)
    
def mult(a,b):
    print("Mult = ", a*b)
    
class Adder:
    def __init__(self,a,b):
        self.a = a
        self.b = b
        
    def add(self):
        print(self.a + self.b)