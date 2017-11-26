using Perceptrons


@testset "Voted Perceptron Tests (in sample)" begin


    @testset "OR function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [1.0 ; 1.0; 1.0; 0.0]

        model = Perceptrons.fit(X,Y,centralize=false,mode="voted")
        pred  = Perceptrons.predict(model,X)

    	@test all(pred .== Y)

        model = Perceptrons.fit(X,Y,centralize=true,mode="voted")
        pred  = Perceptrons.predict(model,X)

        @test all(pred .== Y)

    end

    @testset "AND function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [1.0 ; 0.0; 0.0; 0.0]

        model = Perceptrons.fit(X,Y,centralize=false,mode="voted")
        pred  = Perceptrons.predict(model,X)

        @test all(pred .== Y)

        model = Perceptrons.fit(X,Y,centralize=true,mode="voted")
        pred  = Perceptrons.predict(model,X)

        @test all(pred .== Y)

    end

end


@testset "Voted Perceptron Tests (out of sample)" begin


    @testset "OR function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [1.0 ; 1.0; 1.0; 0.0]
        Xt = X .+ .3
        model = Perceptrons.fit(X,Y,centralize=true,mode="voted")
        pred  = Perceptrons.predict(model,Xt)
        @test all(pred .== Y)

    end

    @testset "AND function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [1.0 ; 0.0; 0.0; 0.0]
        Xt = X .+ .03

        model = Perceptrons.fit(X,Y,centralize=true,alpha=1.0,mode="voted")
        pred  = Perceptrons.predict(model,Xt)

    end

end

@testset "Check Labels" begin

    X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
    Y = [1.0 ; -1; 0.0; 0.0]

    try model = Perceptrons.fit(X,Y,mode="voted") catch @test true end

end
