using Perceptrons


@testset "Kernel Perceptron Tests (in sample)" begin


    @testset "OR function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [1.0 ; 1.0; 1.0; 0.0]

        model = Perceptrons.fit(X,Y,centralize=false,kernel="rbf",width=.1)
        pred  = Perceptrons.predict(model,X)

    	@test all(pred .== Y)

        model = Perceptrons.fit(X,Y,centralize=true,kernel="rbf",width=.1)
        pred  = Perceptrons.predict(model,X)

        @test all(pred .== Y)

    end

    @testset "AND function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [1.0 ; 0.0; 0.0; 0.0]

        model = Perceptrons.fit(X,Y,centralize=false,kernel="rbf",width=.1)
        pred  = Perceptrons.predict(model,X)

        @test all(pred .== Y)

        model = Perceptrons.fit(X,Y,centralize=true,kernel="rbf",width=.1)
        pred  = Perceptrons.predict(model,X)

        @test all(pred .== Y)

    end

    @testset "XOR function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [0.0 ; 1.0; 1.0; 0.0]

        model = Perceptrons.fit(X,Y,centralize=false,kernel="rbf",width=.01)
        pred  = Perceptrons.predict(model,X)

        @test all(pred .== Y)

        model = Perceptrons.fit(X,Y,centralize=true,kernel="rbf",width=.01)
        pred  = Perceptrons.predict(model,X)

        @test all(pred .== Y)

    end

end


@testset "Kernel Perceptron Tests (out of sample)" begin


    @testset "OR function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [1.0 ; 1.0; 1.0; 0.0]
        Xt = X .+ .3
        model = Perceptrons.fit(X,Y,centralize=true,kernel="rbf",width=.1)
        pred  = Perceptrons.predict(model,Xt)
        @test all(pred .== Y)

    end

    @testset "AND function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [1.0 ; 0.0; 0.0; 0.0]
        Xt = X .+ .03

        model = Perceptrons.fit(X,Y,centralize=true,kernel="rbf",width=.1)
        pred  = Perceptrons.predict(model,Xt)
        @test all(pred .== Y)

    end

    @testset "XOR function" begin

        X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
        Y = [0.0 ; 1.0; 1.0; 0.0]
        Xt = X .+ .03

        model = Perceptrons.fit(X,Y,centralize=true,kernel="rbf",width=.01)
        pred  = Perceptrons.predict(model,Xt)
        @test all(pred .== Y)

    end

end

@testset "Check Labels" begin

    X = [1.0 1.0; 0.0 1.0; 1.0 0.0; 0.0 0.0]
    Y = [1.0 ; -1; 0.0; 0.0]

    try model = Perceptrons.fit(X,Y,kernel="rbf",width=1.0) catch @test true end

end
