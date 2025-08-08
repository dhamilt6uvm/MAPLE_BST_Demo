using PyCall
using SparseArrays
using Plots
gr()
using ColorTypes
using Colors
using JuMP
using Ipopt
using LinearAlgebra
using CSV
using DataFrames

##### Function for running power flow ########
function solve_pf(psm::PyObject, V0_ref::Vector{ComplexF64}, t_ind::Int64, linear_solver::String)

    n_nodes = length(psm.Nodes)
    n_branches = length(psm.Branches)

    # Model Setup
    model = Model(Ipopt.Optimizer)
    if linear_solver in ["ma27","ma57","ma77","ma86","ma97"]
        set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        set_attribute(model, "linear_solver", linear_solver)
    elseif linear_solver == "mumps"
        set_attribute(model, "linear_solver", linear_solver)
    else
        throw(ArgumentError("linear_solver $linear_solver not supported."))
    end
    set_optimizer_attribute(model, "print_level", 0)

    # Variable Definitions
    @variable(model, Vph_real[ph=1:3,1:n_nodes], start=real(V0_ref[ph]*exp(-im*pi/6)))
    @variable(model, Vph_imag[ph=1:3,1:n_nodes], start=imag(V0_ref[ph]*exp(-im*pi/6)))
    @variable(model, Iph_real[1:3,1:n_branches], start=0)
    @variable(model, Iph_imag[1:3,1:n_branches], start=0)

    set_start_value.(Vph_real[:,1], real(V0_ref))
    set_start_value.(Vph_imag[:,1], imag(V0_ref))

    # Complex Variable Expressions
    @expression(model, Vph, Vph_real.+im*Vph_imag)
    @expression(model, Iph, Iph_real.+im*Iph_imag)

    # Substation Voltage Constraint
    @constraint(model, Vph[:,1] .== V0_ref)

    # Power Flow Constraints
    pb_lhs = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
    pb_rhs = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
    for (br_ind,Branch) in enumerate(psm.Branches)
        # skip open branches
        if Branch.type == "switch" 
            if Branch.status == "OPEN"
                Iph[:,br_ind] .== 0.0
                continue
            end
        end
        from_node_ind = Branch.from_node_ind+1
        to_node_ind = Branch.to_node_ind+1
        # "Ohm's law"
        @constraint(model, Vph[:,from_node_ind] .== Branch.A_br*Vph[:,to_node_ind] + Branch.B_br*Iph[:,br_ind])
        # Add branch flows to power balance expressions
        pb_lhs[:,to_node_ind] += diag(Vph[:,to_node_ind]*Iph[:,br_ind]')
        pb_rhs[:,from_node_ind] += diag(Branch.A_br*Vph[:,to_node_ind]*Iph[:,br_ind]'*(Branch.D_br')+Branch.B_br*Iph[:,br_ind]*Iph[:,br_ind]'*(Branch.D_br'))
    end


    # Power Injection Constraints
    s_load = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
    for (ld_ind, Load) in enumerate(psm.Loads)
        if haskey(Load,"Sload")
            s_load[:,Load.parent_node_ind+1] += Load.Sload[t_ind,:]
        end
    end
    s_gen = zeros(GenericQuadExpr{ComplexF64, VariableRef}, 3, n_nodes)
    for (gen_ind, Gen) in enumerate(psm.Generators)
        if haskey(Gen,"Sgen")
            s_gen[:,Gen.parent_node_ind+1] += Gen.Sgen[t_ind,:]
        end
    end
    for (sht_ind, Shunt) in enumerate(psm.Shunts)
        if Shunt.type == "capacitor"
            status = zeros(Int, 3, 1)
            if Shunt.switchA == "CLOSED"
                status[1] = 1
            end
            if Shunt.switchB == "CLOSED"
                status[2] = 1
            end
            if Shunt.switchC == "CLOSED"
                status[3] = 1
            end
            parent_node_ind = Shunt.parent_node_ind+1
            s_load[:,parent_node_ind] += status.*diag(Vph[:,parent_node_ind]*Vph[:,parent_node_ind]'*conj(Shunt.Ycap))
        end
    end
    @constraint(model, pb_rhs[:,2:end] - pb_lhs[:,2:end] .== s_gen[:,2:end]-s_load[:,2:end])

    optimize!(model)

    # print status
    status = termination_status(model)
    if status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        print(".")
    else
        println("Solver did not find an optimal solution: $status")
    end

    return value.(Vph)#, value.(pb_rhs[:,1]-pb_lhs[:,1])

end