module AllFuncs

using PyCall
const hasattr = pyimport("builtins").hasattr

export get_loadgen_nodes_LO, get_netload_onetime, get_is_day, get_gen_idx, write_netload_onetime

function get_loadgen_nodes_LO(psm)
    # returns list of unique nodes that have an Sload or Sgen attribute - 1-indexed! 
    # returns length of that list
    # note: (L)OAD (O)RDER node numbers are in the order they appear in when looping "for load in psm.Loads", then again with gens
    nodes = Int[]               # unique nodes with a load or a gen
    # Extract nodes with load/gen/both
    for load in psm.Loads
        if hasattr(load, :Sload)
            ind = load.parent_node_ind + 1          # parent_node_ind is 0-indx add 1
            if !(ind in nodes)
                push!(nodes, ind)
            end
        end
    end
    for gen in psm.Generators
        if hasattr(gen, :Sgen)
            ind = gen.parent_node_ind + 1           # parent_node_ind is 0-indx add 1
            if !(ind in nodes)
                push!(nodes, ind)
            end
        end
    end
    n_loads = length(nodes)
    return nodes, n_loads
end

function get_netload_onetime(psm, nodes, t_ind, ph_col)
    n_loads = length(nodes)
    Sload_tind = zeros(ComplexF64, n_loads)
    for (ii,node) in enumerate(psm.Nodes[nodes])            # loop over all supplied nodes (in supplied order) - need plus one
        tmp = 0
        for load_ind in node.loads                          # loop over all loads at that node
            load = psm.Loads[load_ind+1]                    # plus one for 0-indx load_ind
            if hasattr(load, "Sload")                       # check that load has an Sload attribute
                tmp += load.Sload[t_ind,ph_col]             # add up loads to temporary
            end                                             # load is positive because load goes into BST positive
        end
        for gen_ind in node.gens                            # loop over all gens at node
            gen = psm.Generators[gen_ind+1]
            if hasattr(gen, "Sgen")
                tmp -= gen.Sgen[t_ind,ph_col]               # subtract gen values from temp
            end                                             # taking a load positive convention.. NEED TO LOOK AT THIS IF THERE ARE JUST GEN NODES
        end
        Sload_tind[ii] = tmp
    end
    return Sload_tind
end

function get_is_day(test_idx, day_hours, night_hours)
    is_day = Bool[]
    for idx in test_idx
        hour = mod1(idx,24)
        if hour in day_hours
            push!(is_day, true)
        elseif hour in night_hours
            push!(is_day, false)
        else
            println("Error, hour not in either set")
        end
    end
    return is_day
end

function get_gen_idx(psm, nodes)
    # get list of indices that have generators
    gen_idx_in_loadgens = []
    for (ii,node) in enumerate(psm.Nodes[nodes])
        if length(node.gens) > 0
            for gen_ind in node.gens
                gen = psm.Generators[gen_ind+1]             # plus one for gen_ind being 0-indx
                if hasattr(gen, "Sgen")
                    push!(gen_idx_in_loadgens, ii)
                end
            end
        end
    end
    return gen_idx_in_loadgens
end

function write_netload_onetime(psm, nodes, t_ind, Sload_all)
    # input Sload is vector at the time index that is being modified
    for (ii,node) in enumerate(psm.Nodes[nodes])            # loop over all supplied nodes (in supplied order) - need plus one
        ct = 0                                              # count for if first load or not
        for load_ind in node.loads                          # loop over all loads at that node
            load = psm.Loads[load_ind+1]
            if hasattr(load, "Sload")                       # check that load has an Sload attribute
                if ct == 0                                  # if haven't overwritten any loads yet
                    load.Sload[t_ind,ph_col] = Sload_all[ii]    # write the load to the Sload value
                else
                    load.Sload[t_ind,ph_col] = 0            # write any other loads to zero
                end
                ct += 1                                     # bump the counter
            end                                             # load is positive because load goes into BST positive
        end
        for gen_ind in node.gens                            # loop over all gens at node
            gen = psm.Generators[gen_ind+1]
            if hasattr(gen, "Sgen")
                gen.Sgen[t_ind,ph_col] = 0                  # write the gen value to 0 
            end                                             # taking a load positive convention.. NEED TO LOOK AT THIS IF THERE ARE NODES W/ JUST GEN
        end
    end
end

end # end module