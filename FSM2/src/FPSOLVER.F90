subroutine FPSOLVER(cv,D_a,D_ave,D_m,fsnow_current)

! This uses a "fixed-point" iteration algorithm to solve for the
!   depth of melt when the cell-average snow depth is known.

    implicit none

    integer :: &
        maxiter,        & ! Maximun number of iterations
        i                 ! Iteration counter

    real, intent(in) :: &
        cv,                 & ! coefficient of variation for the subgrid snow variation
        D_a,                & ! sum of all snow accumulation events prior to this time
        D_ave                 ! cell-average snow depth

    real, intent(inout) ::  &
        D_m,               & ! sum of all snowmelt events prior to this time           
        fsnow_current               ! snow covered area fraction (for a given total melt depth, D_m)

    real :: &
        D_m_old,           &  ! D_m initial guess
        zeta,              &  ! 
        xlambda,           &  !
        z_Dm,              &  !
        tol                   ! melt-depth tolerance

! Define the melt-depth tolerance to be 0.1 mm.
    tol = 1.0e-4
    maxiter = 10

! Set the initial guess to a small number (this does not seem to
! affect the iterations required to gain convergence, and
! starting with a large number can lead to divergence of the
! solution).
    D_m_old = 1.0e-9

    zeta = sqrt(log(1.0 + max(cv,0.001)**2))
    xlambda = log(D_a) - 0.5 * zeta**2

    do i=1,maxiter
        z_Dm = (log(D_m_old) - xlambda) / zeta
        fsnow_current = 0.5 * erfc(z_Dm/sqrt(2.0))
        D_m = (0.5 * exp(xlambda + 0.5*zeta**2) * erfc((z_Dm - zeta)/sqrt(2.0)) - D_ave) / fsnow_current

        if (abs(D_m - D_m_old) > tol) then
            D_m_old = D_m
        end if
    enddo

end subroutine FPSOLVER