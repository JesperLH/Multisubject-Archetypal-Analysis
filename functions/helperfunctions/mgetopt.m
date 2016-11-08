% Parser for optional arguments
function var = mgetopt(opts, varname, default, varargin)
    if isfield(opts, varname)
        var = getfield(opts, varname); %#ok<GFLD>
    else
        var = default;
    end
end