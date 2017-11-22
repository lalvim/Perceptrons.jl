
@testset "Linear Perceptron Tests (in sample)" begin


    @testset "OR function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [1.0 ; 1.0; 1.0; 0.0]

        model = Perceptrons.fit(X,Y)
        pred  = Perceptrons.predict(model,X)

    	@test all(pred .== Y)
    end

    @testset "AND function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [1.0 ; 0.0; 0.0; 0.0]

        model = Perceptrons.fit(X,Y)
        pred  = Perceptrons.predict(X)

    	@test all(pred .== Y)
    end

end


@testset "Linear Perceptron Tests (out of sample)" begin


    @testset "OR function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [1.0 ; 1.0; 1.0; 0.0]
        Xt = X .+ .1
        model = Perceptrons.fit(X,Y)
        pred  = Perceptrons.predict(model,Xt)

    	@test all(pred .== Y)
    end

    @testset "AND function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [1.0 ; 0.0; 0.0; 0.0]
        Xt = X .+ .1

        model = Perceptrons.fit(X,Y)
        pred  = Perceptrons.predict(Xt)

    	@test all(pred .== Y)
    end

end

@testset "Check Labels (must be {0,1})" begin

    X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
    Y = [1.0 ; -1; 0.0; 0.0]

    try model = Perceptrons.fit(X,Y) catch @test true end

end
