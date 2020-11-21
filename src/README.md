# Core part of deGraphcS
configs.py            --Config the hyper-parameters of deGraphCS

dataloader.py         --Load the data in batch

util_IR.py            --Preprocess the origin IR to generate grpahs, which can be identified by graph neural networks

util_desc.py          --Preprocess the comments

generate_interface.py --Generate the interfaces of the third-party libraries

# Generate the interfaces to solve the compilation probelm
## An example to show how the compilation problem of IR can be solved 
### Initial code snippets crawled from Github
```
public void range(IHypercube space, IvisitKDNode visitor){
   if(root == null) return;
   root.getRange(space, visitor);
}
```
The code above cannot be compiled because of the following parts:
1. The third-library IHypercube and IvisitKDNode.
2. The object root and its method getRange.

The third-library missing probelm can be solved by adding some empty interfaces (the Root class with getRange method, IHypercube class and IvisitKDNode class) since the realization details of the method are not neccessary. 
### After adding the interface, the example source code can be successfully compiled:
```
public class Range{
    private Root root;
    public void range(IHypercube space, IvisitKDNode visitor){
        if(root == null) return;
        root.getRange(space, visitor);
    }
}
class Root{
   public void getRange(IHypercube space, IvisitKDNode visitor){
      return;
   }
}
class IHypercube{}
class IvisitKDNode{}
```
