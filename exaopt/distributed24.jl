using Test
using ProxAL
using DelimitedFiles, Printf
using LinearAlgebra, JuMP, Ipopt
using ExaOpt
using CatViews
using MPI

MPI.Init()
DATA_DIR = joinpath(dirname(@__FILE__), "..", "data")
case = "case1354pegase"
T = 24
K = 0
ramp_scale = 0.30
load_scale = 0.1
maxρ = 0.001
quad_penalty = 1000
rtol = 1e-4


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
algparams.verbose = 1
algparams.decompCtgs = false
algparams.device = ProxAL.CPU
algparams.iterlim = 100
# Tolerance of the Newton-Raphson algorithm
algparams.nr_tol = 1e-10
# algparams.optimizer =
# optimizer_with_attributes(Ipopt.Optimizer, "print_level" => Int64(algparams.verbose > 0)*5)

algparams.optimizer = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "limited_memory_max_history" => 50,
        "hessian_approximation" => "limited-memory",
        "derivative_test" => "first-order",
        "tol" => 1e-6,
)
algparams.gpu_optimizer = ExaOpt.AugLagSolver(; max_iter=20, ωtol=1e-4, verbose=1, α0=0.00001, inner_algo=:tron)

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
runinfo = run_proxALM(opfdata, rawdata, modelinfo, algparams)
MPI.Finalize()
