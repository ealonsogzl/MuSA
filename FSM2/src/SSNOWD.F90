subroutine SSNOWD(cv,D_ave,D_a,D_m,fsnow,snet)
    
    implicit none
    real, intent(in) :: &
        cv,             & ! coefficient of variation for the subgrid snow variation
        snet              ! net accumulation/melt

    real, intent(inout) :: &
        D_a,            & ! sum of all snow accumulation events prior to this time,
        D_m,            & ! sum of all snowmelt events prior to this time,
        D_ave,           & ! cell-average snow depth
        fsnow             ! snow covered area fraction (for a given total meltdepth, D_m)

    real :: &
        zeta, D_a_current, D_m_current, fsnow_current, D_ave_current, xlambda, z_Dm, safe_snet
    

    ! Compute the net snow gain or loss for this time step.
    !   Don't let the model accumulate extremely small snow falls (e.g.,
    !   the case where dD_a = 2.330483e-07 and dD_m = 2.330483e-07,
    !   but where snet = dD_a - dD_m = +1.421085e-14).  But do let it
    !   include extremely small melts, so that it can get rid of the
    !   accumulations.
        !safe_snet = dD_a - dD_m

        if (snet > 0.0  .and.  snet < 1.0e-7) then
            safe_snet = 0.0
        else
            safe_snet = snet
        endif

    ! Define the required constants.  Don't let anybody define zeta
    !   to be zero, to avoid the divide by zero later on.  To do this
    !   don't let cv be less than 0.001.
        zeta = sqrt(log(1.0 + max(cv,0.001)**2))
        
    ! No snow at previous time step.
        if (D_a == 0.0) then

    ! We have accumulation.
            if (safe_snet > 0.0) then
                D_a_current = safe_snet
                D_m_current = 0.0
                fsnow_current = 1.0
                D_ave_current = D_a_current

    ! No change (or melt with no snow!).
            else
                D_a_current = 0.0
                D_m_current = 0.0
                fsnow_current = 0.0
                D_ave_current = 0.0
            endif

    ! Snow on ground at previous time step.
        else
    
    ! We have accumulation.
            if (safe_snet > 0.0) then
    ! 100% snowcover case at previous time step, with new accumulation.
                if (D_m == 0.0) then
                    D_a_current = D_a + safe_snet
                    D_m_current = 0.0
                    fsnow_current = 1.0
                    D_ave_current = D_a_current

    ! Less than 100% snowcover case at previous time step, with new
    !   accumulation.
    ! There are two ways to deal with this case:
    !   1) Assume that new snow falls over the entire grid cell, giving
    !        that cell an average snow depth equal to the new snow plus
    !        the cell-average snow depth from the fractional snow cover
    !        at the previous time step.  This captures our assumption
    !        that new snow precipitation is distributed over the entire
    !        grid cell, but it is unrealistic because it does not allow
    !        for a small snow accumulation occurring in the middle of
    !        the melt period, to melt off quickly and expose the
    !        previously exposed surface.
    !   2) Use the new accumulation to decrease the melt depth values,
    !        effectively "pushing" the depletion curve back towards 100%
    !        snowcover.  If there is any snow left after the melt depth
    !        has been reduced to zero, add the remaining snow to the
    !        total snow depth.  This represents the ablation period
    !        more realistically, but does not account for the snow
    !        accumulating over the entire grid cell.
    ! So far my preference is to use option 2).  Note that for the case
    !   of melt periods that have occasional snow accumulations,
    !   option 2) generally produces smaller snow-covered fractions
    !   than option 1).  A down-side to option 2) is that I have not
    !   found an analytical solution to this problem, and have had
    !   to use an iterative solution (that fortunately converges very
    !   quickly).
    ! You can comment/uncomment the following to pick option 1) or 2).
                else
! Option 1.
                   !D_a_current = D_ave + safe_snet
                   !D_m_current = 0.0
                   !fsnow_current = 1.0
                   !D_ave_current = D_a_current
! Option 2.
!   Here there are two cases: where there is enough new accumulation
!   to push D_m to zero, and where there is not (D_m still non-zero).
                   if (D_ave + safe_snet >= D_a) then
                       D_a_current = D_ave + safe_snet
                       D_m_current = 0.0
                       fsnow_current = 1.0
                       D_ave_current = D_a_current
                   else
                       D_a_current = D_a
                       D_ave_current = D_ave + safe_snet
                       call FPSOLVER(cv,D_a_current,D_ave_current, &
                                   D_m_current,fsnow_current)
                   endif
                endif
! We have melt (D_m can be >= 0.0).
            elseif (safe_snet < 0.0) then
                D_a_current = D_a
                D_m_current = D_m - snet
! Compute the snow-covered fraction under the assuption of the erf
!   distribution curve.
                xlambda = log(D_a_current) - 0.5 * zeta**2
                z_Dm = (log(D_m_current) - xlambda) / zeta
                fsnow_current = 0.5 * erfc(z_Dm/sqrt(2.0))

! Compute the average snow depth in this grid cell.
                D_ave_current = 0.5 * exp(xlambda + 0.5*zeta**2) * &
                    erfc((z_Dm - zeta)/sqrt(2.0)) - &
                    D_m_current * fsnow_current

! When the computation reduces the snowcover to (near) zero, reset
!   the melt and accumulation variables to zero.
                    if (fsnow_current < 0.005) then
                        D_a_current = 0.0
                        D_m_current = 0.0
                        fsnow_current = 0.0
                        D_ave_current = 0.0
                    endif

! We have no change since the last time step (snet = 0.0).
                else
                    D_a_current = D_a
                    D_m_current = D_m
                    fsnow_current = fsnow
                    D_ave_current = D_ave
                endif
            endif

! Update the values to be used for the next time step.
      D_a = D_a_current
      D_m = D_m_current
      fsnow = fsnow_current
      D_ave = D_ave_current

    end subroutine SSNOWD







