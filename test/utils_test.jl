using Statistics


@testset "Auxiliary Functions Test" begin

    @testset "check constant columns" begin

		try
			Perceptrons.check_constant_cols([1.0 1; 1 2;1 3])
		catch
			@test true
		end

		try
			Perceptrons.check_constant_cols([1.0 1 1][:,:])
		catch
			@test true
		end

		try
			Perceptrons.check_constant_cols([1.0 2 3][:,:])
		catch
			@test true
		end

		try
			Perceptrons.check_constant_cols([1.0; 1; 1][:,:])
		catch
			@test true
		end

		@test Perceptrons.check_constant_cols([1.0 1;2 2;3 3])
		@test Perceptrons.check_constant_cols([1.0;2;3][:,:])

	end

	@testset "centralize" begin

		X        = [1; 2 ;3.0][:,:]
		X        = Perceptrons.centralize_data(X,mean(X,dims=1),std(X,dims=1))
		@test all(X .== [-1,0,1.0])

	end

	@testset "decentralize" begin

		Xo        = [1; 2; 3.0][:,:]
		Xn        = [-1,0,1.0][:,:]
		Xn        = Perceptrons.decentralize_data(Xn,mean(Xo,dims=1),std(Xo,dims=1))
		@test all(Xn .== [1; 2; 3.0])

	end

	@testset "checkparams" begin

         try
			 Perceptrons.check_params("linear")
		 catch
			 @test true
		 end
		 try
			 Perceptrons.check_params("x")
		 catch
			 @test true
		 end

	end

	@testset "checkdata" begin

		 try
			 Perceptrons.check_data(zeros(0,0), 0)
		 catch
			 @test true
		 end
		 try
			 Perceptrons.check_data(zeros(1,1), 10)
		 catch
			 @test true
		 end
		 @test Perceptrons.check_data(ones(1,1), 1)

	end

	@testset "check binary labels" begin

		try
			Perceptrons.check_linear_binary_labels([1,0,2])
		catch
			@test true
		end
		try
			Perceptrons.check_linear_binary_labels([1,-1])
		catch
			@test true
		end

		@test Perceptrons.check_linear_binary_labels([1,0])

	end

end;
