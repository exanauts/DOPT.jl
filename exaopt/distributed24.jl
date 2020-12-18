using Test
using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP, Ipopt
using ExaOpt
using CatViews
using MPI

MPI.Init()
DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case = "case9"
T = 24
K = 0
ramp_scale = 0.30
load_scale = 0.85
maxρ = 0.001
quad_penalty = 1000
rtol = 1e-4

if case == "case9"
        T = 2
        ramp_scale = 0.5
        load_scale = 1.0
        maxρ = 0.1
        quad_penalty = 0.1
end

if case == "case118"
        T = 24
        ramp_scale = 0.20
        load_scale = 1.0
        maxρ = 1.0
        quad_penalty = 1e5
end

# Load case
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, "mp_demand", "$(case)_oneweek_168")
rawdata = RawData(case_file, load_file)
ctgs_arr = deepcopy(rawdata.ctgs_arr)

# Model/formulation settings
modelinfo = ModelParams()
modelinfo.num_time_periods = T
modelinfo.load_scale = load_scale
modelinfo.ramp_scale = ramp_scale
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = false
modelinfo.weight_quadratic_penalty_time = quad_penalty
modelinfo.weight_freq_ctrl = quad_penalty
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :frequency_ctrl
modelinfo.case_name = case
modelinfo.num_ctgs = K

# Algorithm settings
algparams = AlgParams()
algparams.parallel = true #algparams.parallel = (nprocs() > 1)
algparams.verbose = 2
algparams.decompCtgs = false
algparams.device = ProxAL.CPU
algparams.iterlim = 100
# Tolerance of the Newton-Raphson algorithm
algparams.nr_tol = 1e-10
algparams.optimizer =
optimizer_with_attributes(Ipopt.Optimizer, "print_level" => Int64(algparams.verbose > 0)*5)

# algparams.optimizer = optimizer_with_attributes(
#         Ipopt.Optimizer,
#         "print_level" => 0,
#         "limited_memory_max_history" => 50,
#         "hessian_approximation" => "limited-memory",
#         "derivative_test" => "first-order",
#         "tol" => 1e-6,
# )
algparams.gpu_optimizer = ExaOpt.AugLagSolver(; max_iter=20, ωtol=1e-4, verbose=1, α0=1e-12, inner_algo=:projectedgradient)

# rawdata.ctgs_arr = deepcopy(ctgs_arr[1:modelinfo.num_ctgs])
opfdata = opf_loaddata(rawdata;
                       time_horizon_start = 1,
                       time_horizon_end = T,
                       load_scale = load_scale,
                       ramp_scale = ramp_scale)
set_rho!(algparams;
         ngen = length(opfdata.generators),
         modelinfo = modelinfo,
         maxρ_t = maxρ,
         maxρ_c = maxρ)

algparams.mode = :coldstart
runinfo = run_proxALM(opfdata, rawdata, modelinfo, algparams, ProxAL.ReducedSpace(); init_opf = true)

#----- plotting -----#
using Plots
using LaTeXStrings

if algparams.verbose > 1 && MPI.Comm_rank(MPI.COMM_WORLD) == 0
        ENV["GKSwstype"]="nul"
        algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 5)
        algparams.mode = :nondecomposed
        result = solve_fullmodel(opfdata, rawdata, modelinfo, algparams)
        xstar = result["primal"]
        zstar = result["objective_value_nondecomposed"]

        algparams.mode = :lyapunov_bound
        result = solve_fullmodel(opfdata, rawdata, modelinfo, algparams)
        lyapunov_star = result["objective_value_lyapunov_bound"]


        function options_plot(plt)
                fsz = 20
                plot!(plt,
                        fontfamily = "Computer-Modern",
                        yscale = :log10,
                        framestyle = :box,
                        ylim = [1e-4, 1e+1],
                        xtickfontsize = fsz,
                        ytickfontsize = fsz,
                        guidefontsize = fsz,
                        titlefontsize = fsz,
                        legendfontsize = fsz,
                        size = (800, 800),
                        legend = :bottomleft
                )
        end

        function initialize_plot()
                gr()
                label= ["|Ramp-error|"
                        "|KKT-error|"
                        "|x-x^*|"
                        "|c(x)-c(x^*)|/c(x^*)"
                        "|L-L^*|/L^*"]
                any = Array{Any, 1}(undef, length(label))
                any .= Any[[1,1]]
                plt = plot([Inf, Inf], any,
                                lab = reshape(label, 1, length(label)),
                                lw = 2.5,
                                # markersize = 2.5,
                                # markershape = :auto,
                                xlabel="Iteration")
                options_plot(plt)
                return plt
        end

        plt = initialize_plot()
        for iter=1:runinfo.iter
                optimgap = 100.0abs(runinfo.objvalue[iter] - zstar)/abs(zstar)
                lyapunov_gap = 100.0(runinfo.lyapunov[iter] - lyapunov_star)/abs(lyapunov_star)
                push!(plt, 1, iter, runinfo.maxviol_t[iter])
                push!(plt, 2, iter, runinfo.maxviol_d[iter])
                push!(plt, 3, iter, runinfo.dist_x[iter])
                push!(plt, 4, iter, optimgap)
                push!(plt, 5, iter, (lyapunov_gap < 0) ? NaN : lyapunov_gap)
        end
        np = MPI.Comm_size(MPI.COMM_WORLD)
        savefig(plt, case * ".plot_$(case)_$(np).png")
end
MPI.Finalize()
