using Distributions
using FFTW

export mean_field_simulation, get_mean_diff, get_order_parameter, detrend, power_analysis


function positive_part(x)
    return max(0, x)
end

# define the BKPZ mean field simulation
function mean_field_simulation(n::Int64, d::Int64, T::Int64, B::Float64)
    taus = Array{Float32}(undef, n, T)
    eps = rand(Exponential(), (n, T))
    taus[:, 1] = eps[:, 1]
    for t in 2:T
        for i in 1:n
            neighbors = rand(1:n, d)
            taus[i, t] = positive_part(maximum(taus[neighbors, t-1]) - B) + eps[i, t]
        end
    end
    return taus
end


function get_mean_diff(taus)
    return vec(mean(diff(taus, dims=2), dims=1))
end

function get_order_parameter(taus)
    return mean(get_mean_diff(taus))
end

function detrend(taus)
    T = size(taus, 2)
    trend = get_order_parameter(taus) * (0:T-1)
    return transpose(transpose(taus) .- trend)
end

function get_fluctuations(taus)
    trend = get_order_parameter(taus) * (0:size(taus, 2)-1)
    fluctuations = vec(mean(taus, dims=1)) .- trend
    return fluctuations
end

function get_power_spectrum(fluctuations)
    F = fftshift(fft(fluctuations))
    t = length(fluctuations)
    F = F[t÷2+1:end]
    return abs.(F)
end

function power_analysis(n::Int64, d::Int64, T::Int64, B::Float64)
    taus = mean_field_simulation(n, d, T, B)
    fluctuations = get_fluctuations(taus)
    power_spectrum = get_power_spectrum(fluctuations)
    return power_spectrum
end