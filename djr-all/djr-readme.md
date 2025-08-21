## Notes for my stuff specifically

00 - genearlly (or 0, or un-numbered) was when I was taking derivative of voltage with respect to injections at every load and every generator on the BH small 02 network (which has 230 loads and 30 generators). However, as I eventually learned, this gets tricky. 
01 - (or 1) now taking derivative with respect to NET injections at each UNIQUE node. There are 209 unique nodes that have a load a gen or both (they all have a load, one of them has 2 loads, and 30 of them have a gen). This means that now everything is 209x209 instead of 1072x240 or something   
  
  
## Breadcrumbs for BST-work
Generally, reference the powerpoint notes from meeting on 8/20
#### Tasks that I left open: 
- evaluating if the optimal value I'm using is actually optimal
    - was doing this by plotting the 2D feasible sets (see vvc-optimal-design/IPopt13.jl)
    - feasbile sets looked different than expected when a subsection of Jq was used instead of the symmetric and highly structured X matrix
    - Next steps: put the optimization formulation in there (matrix 2-norm only?) and see if the optimal solution is actually inside the set
    - Need to confirm that the two nodes I picked to pull out Jq values are actually next to each other (need to get better network mapping in general)
- linear approximation of voltages with VVC
    - shown in the voltages plot from optimize-kvals.jl
    - voltage at nodes with big inverter slopes look WAY high compared to the other nodes (and they flipped from below 1 to above 1 which shouldn't happen?)
    - possible that this plot IS realistic, but need to prove to myself that's the case
    - would be good to verify in some way OTHER than the linearized model, but that's tricky
- BST & VVC in a loop
    - code in optimize-kvals-solveBSTpf.jl
    - a few funky things happening:
        - power flow returning infeasible solution: may need to study maximum power flow restrictions - Glover section 5.5 - these will cause BST to say infeasible
        - in the current set up, the voltage converges back to exactly what it was before VVC, which doesn't make sense. Have only observed this at one particular K value but should confirm that it happens on multiple
        - had to curtail the highest k value to be way smaller just so that things wouldn't go unstable immediately - not a good sign
    - not sure how to deal with demand of Q, probably should just leave it
    - Next steps: 
        - figure out the linear model first probably (previous section)
        - try not adding Q via VVC but just add some Q and see how voltages change $\to
        $ confirm that the network responds to Q as expected
        - hopefully learn something from those things to then fix the above funkyness
- Optimize over the spectral radius constraint
    - Hadn't really made any progress on this. Will thought he had a shortcut but it was fake news
    - Have some notes on potential ways to solve this from 8/13 or 8/6 meeting notes in powerpoint
    - It's a nice to have, but almost a need to have

#### Directions:
Goal was to repeat the figures from matlab with grouped bar chart: For the 10 hours of the year with the most voltage deviation, given my optimal design of slopes, how much better would voltage be? Need to get BST&VVC in loop for this. 

What results I need to replicate will become more clear after writing the paper with MATLAB network and results