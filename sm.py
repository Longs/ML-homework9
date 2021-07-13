from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        # Your code here
        
        s = self.start_state
        output = []
        
        for x in input_seq:
            s =self.transition_fn(s,x)
            output.append(self.output_fn(s))
        return output


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    #least significant to most significant add and carry
    start_state = (0,0) # (output this digit, carry next digit)

    def transition_fn(self, s, x):
        
        next_carry = 0
        #current carry will move to previous slot
        s0,s1 = s
        x0,x1 = x
        sum = s1+x0+x1
        if sum == 0:
            s = (0,0)
        if sum == 1:
            s = (1,0)
        if sum == 2:
            s = (0,1)
        if sum ==3:
            s = (1,1)
        return s

    def output_fn(self, s):
        s1,s2 = s
        return s1


class Reverser(SM):
    start_state = [False,[]] #whether end has been seen and list to reverse

    def transition_fn(self, s, x):
        # Your code here
        if x == 'end':
            s[0] = True
        else:
            if s[0] == False:
                s[1].append(x)
        return s
        

    def output_fn(self, s):
        if s[0] and len(s[1]) > 0:
            return s[1].pop()
        else:
            return None



class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        # Your code here
        pass

    def transition_fn(self, s, i):
        # Your code here
        pass

    def output_fn(self, s):
        # Your code here
        pass

sm = Accumulator()

print("Output should = [-1, 1, 4, 2, 7, 13]")
print(sm.transduce([-1, 2, 3, -2, 5, 6]))

print(Reverser().transduce(list('the') + ['end'] + list(range(3))))

#2.2.c
#lists in the firm v0 = [v0+v1+v2+v3]
orig_lists = [[0,0.09,0.81,0],[0.81,0.09,0,0],[0,0,0.09,0.81],[0.81,0,0,0.09]]
orig_matrix = np.matrix([[0,0.09,0.81,0],[0.81,0.09,0,0],[0,0,0.09,0.81],[0.81,0,0,0.09]])
print(orig_matrix)

for _ in range(len(orig_lists)):
    orig_lists[_][_] = orig_lists[_][_] -1

A = np.matrix(orig_lists)
b = np.matrix([[0],[-1],[0],[-2]])

print(np.linalg.solve(A,b))