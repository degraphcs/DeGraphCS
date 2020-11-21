# Online-Repo

## An example to show the recursive call and loop functions can be represented as close as possible from our variable-based flow graph
### An example "get sum" function realized by recursive call
```
int get_sum(int N){
    int sum = 0;
    while(N != 0){
       sum += N;
       N-=1;
    }
    return sum;
}
```
### An example "get sum" function realized by loop functions
```
int get_sum(int N){
    if(N == 0){return N;}
    else{
       int sum;
       sum = N + get_sum(N-1);
       return sum; 
    }
}
```
### The corresponding generated variable-based flow graphs are shown as below:

### To better illustrate the common charateristics of variable-based flow graph constructed by deGraphCS from the above two different realizations
<img src="https://github.com/xxx-ano/Online-Repo/blob/main/example_recursive.jpg" width="300" height="300" alt="the constructed graph"/><br/>

## The details of the equations and algorithms in deGraphCS
### The realization details of the attention mechanism on the whole graph and the comments
```
self_attn = nn.Linear(self.n_hidden, self.n_hidden)
self_attn_scalar = nn.Linear(self.n_hidden, 1)
```

Here, function f() in Equation (2) and Equation (4) means the first MLP layer: nn.Linear(self.n_hidden, self.n_hidden).

u_vfg means the second MLP layer: nn.Linear(self.n_hidden, 1), which can be seen as a high level representation of the VFG nodes.

h_vfg means the final weighted sum embedding of the whole graph (the weighted sum of self_attn_scalar and each node's final embedding). The difference between u_vfg and h_vfg is the same for the corresponding part of Equation (4) and Equation (5).

### The aggragation function used in Equation (1)
The aggregation function used in Equation (1) can be illustrated as follows:
<img src="https://github.com/xxx-ano/Online-Repo/blob/main/propagation%20model.png" width="800" height="120" alt="propogation model"/><br/>

In the functions above, Eq. 1 is the initialization step, which copies node annotations into the first components
of the hidden state and pads the rest with zeros. 

Eq. 2 is the step that passes information between
different nodes of the graph via incoming and outgoing edges with parameters dependent on the edge
type and direction. 

The remaining are GRU-like updates that incorporate information from the other nodes and from the previous timestep
to update each node’s hidden state.









