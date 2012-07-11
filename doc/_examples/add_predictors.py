def add_predictors(base_formula, extra_predictors):
    # Interpret formula in caller's environment:
    env = EvalEnvironment.capture(1)
    desc = ModelDesc.from_formula(base_formula, env)
    # Using LookupFactor here ensures that everything will work correctly even
    # if one of the column names in extra_columns is named like "weight.in.kg"
    # or "sys.exit()" or "LittleBobbyTables()".
    desc.rhs_termlist += [Term([LookupFactor(p)]) for p in extra_predictors]
    return desc
