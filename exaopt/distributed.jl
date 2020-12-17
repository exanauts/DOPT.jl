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
T = 2
K = 0
ramp_scale = 0.5
load_scale = 1.0
maxρ = 0.1
quad_penalty = 0.1
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
algparams.device = ProxAL.CUDADevice
algparams.iterlim = 20
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

# To use RGM instead, change inner_algo to :projectedgradient
algparams.gpu_optimizer = ExaOpt.AugLagSolver(; max_iter=20, ωtol=1e-4, verbose=1, inner_algo=:tron)

@testset "Test ProxAL on $(case) with $T-period, $K-ctgs, time_link=penalty and Ipopt" begin

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
    @test isapprox(runinfo.maxviol_c[end], 0.0)
    @test isapprox(runinfo.x.Pg[:], [0.8979849196165037, 1.3432106614001416, 0.9418713794662078, 0.9840203268799962, 1.4480400989162827, 1.0149638876932787], rtol = rtol)
    # @test isapprox(runinfo.λ.ramping[:], [0.0, 0.0, 0.0, 2.1600093405682597e-6, -7.2856620728201185e-6, 5.051385899057505e-6], rtol = rtol)
    # @test isapprox(runinfo.maxviol_t[end], 2.687848059435005e-5, rtol = rtol)
    # @test isapprox(runinfo.maxviol_d[end], 7.28542741650351e-6, rtol = rtol)
    @test runinfo.iter == 5
end
MPI.Finalize()
