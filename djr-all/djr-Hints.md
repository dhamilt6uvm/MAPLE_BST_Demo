## Hints for working with Julia / this library

### Command line things:  
- Show the elements in a pyobject:  
`py"dir"(psm.Branches[1])`  

- Show elements of a larger thing...   like the psm  
`psm.<tab>`   

- Get basic info on a psm  
`psm`   
  
- See type of defined variable  
`typeof(x)`   

#### For general help  
- To question what different commands do using Julia's built in stuff, in REPL type  
`?`  

- Find names of functions in a package using  
`names(pkg)`  

- Common packages with names:  
    - Base — core functions and types (already loaded)
    - Core — lower-level language internals
    - LinearAlgebra — matrices, norms, factorizations, etc.
    - SparseArrays — sparse matrix support
    - Random — random number generation
    - Statistics — mean, std, etc.
    - Dates — date/time handling
    - Printf — formatted printing
    - DelimitedFiles — reading/writing text data (like CSV)
    - Serialization — saving/loading Julia objects

#### Manipulating the terminal
- Enter help mode: `?`  
- Download packages: ...
- To escape any mode, use backspace

